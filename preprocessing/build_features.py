import os, argparse, numpy as np
import tifffile as tiff
from scipy.ndimage import gaussian_laplace, gaussian_filter, gaussian_gradient_magnitude

# -------------------- I/O --------------------

def load_stack(src, z_start, z_count):
    """Return a small z-stack [D,H,W] from a dir of TIFF slices or a multi-page TIFF."""
    if os.path.isdir(src):
        files = sorted([f for f in os.listdir(src) if f.lower().endswith(('.tif', '.tiff'))])
        if not files: raise FileNotFoundError(f"No .tif files in {src}")
        z_end = min(len(files), z_start + z_count)
        arrs = [tiff.imread(os.path.join(src, files[z])) for z in range(z_start, z_end)]
        vol = np.stack(arrs, axis=0)
    else:
        with tiff.TiffFile(src) as tf:
            n_pages = len(tf.pages)
            z_end = min(n_pages, z_start + z_count)
            vol = np.stack([tf.pages[z].asarray() for z in range(z_start, z_end)], axis=0)
    return vol.astype(np.float32)  # [D,H,W]

def crop_center(vol, z, y, x, D, H, W):
    """Center a patch on (z,y,x) with clipping at boundaries."""
    Dz, Hy, Wx = vol.shape
    z0 = max(0, z - D//2); z1 = min(Dz, z0 + D); z0 = z1 - D
    y0 = max(0, y - H//2); y1 = min(Hy, y0 + H); y0 = y1 - H
    x0 = max(0, x - W//2); x1 = min(Wx, x0 + W); x0 = x1 - W
    z0, y0, x0 = max(0,z0), max(0,y0), max(0,x0)
    patch = vol[z0:z1, y0:y1, x0:x1]
    mask = np.ones_like(patch, dtype=np.float32)
    return patch, mask, (z0,y0,x0)

def robust_clip_norm(a, lo=0.5, hi=99.5, eps=1e-6):
    lo_v, hi_v = np.percentile(a, [lo, hi])
    a = np.clip(a, lo_v, max(hi_v, lo_v + 1e-6))
    m, s = a.mean(), a.std()
    return (a - m) / (s + eps)

# -------------------- Features --------------------

def hessian_eigs(vol, sigma=1.0):
    """Compute Hessian eigenvalues λ1≥λ2≥λ3 per voxel using Gaussian second derivatives."""
    # second derivs
    Hxx = gaussian_filter(vol, sigma=sigma, order=(0,0,2))
    Hyy = gaussian_filter(vol, sigma=sigma, order=(0,2,0))
    Hzz = gaussian_filter(vol, sigma=sigma, order=(2,0,0))
    Hxy = gaussian_filter(vol, sigma=sigma, order=(0,1,1))
    Hxz = gaussian_filter(vol, sigma=sigma, order=(1,0,1))
    Hyz = gaussian_filter(vol, sigma=sigma, order=(1,1,0))
    D, H, W = vol.shape
    # stack to 3x3 Hessian and eigendecompose per voxel
    eigs = np.zeros((3, D, H, W), dtype=np.float32)
    # vectorized-ish: flatten spatial dims for eigh
    Hm = np.stack([Hxx, Hxy, Hxz, Hxy, Hyy, Hyz, Hxz, Hyz, Hzz], axis=0).reshape(3,3,-1)
    # eigh on each column
    vals = []
    for i in range(Hm.shape[-1]):
        M = np.array([[Hm[0,0,i], Hm[0,1,i], Hm[0,2,i]],
                      [Hm[1,0,i], Hm[1,1,i], Hm[1,2,i]],
                      [Hm[2,0,i], Hm[2,1,i], Hm[2,2,i]]], dtype=np.float32)
        w, _ = np.linalg.eigh(M)          # ascending
        vals.append(w[::-1])              # descending
    vals = np.stack(vals, axis=1).reshape(3, D, H, W)
    return vals  # [3,D,H,W] with λ1≥λ2≥λ3

def structure_tensor_orientation(vol, sigma_grad=1.0, sigma_smooth=2.0):
    """Dominant orientation via structure tensor (eigenvector of largest eigenvalue)."""
    # gradients
    gx = gaussian_filter(vol, sigma=sigma_grad, order=(0,0,1))
    gy = gaussian_filter(vol, sigma=sigma_grad, order=(0,1,0))
    gz = gaussian_filter(vol, sigma=sigma_grad, order=(1,0,0))
    # outer products, then smooth
    Jxx = gaussian_filter(gx*gx, sigma=sigma_smooth)
    Jyy = gaussian_filter(gy*gy, sigma=sigma_smooth)
    Jzz = gaussian_filter(gz*gz, sigma=sigma_smooth)
    Jxy = gaussian_filter(gx*gy, sigma=sigma_smooth)
    Jxz = gaussian_filter(gx*gz, sigma=sigma_smooth)
    Jyz = gaussian_filter(gy*gz, sigma=sigma_smooth)
    D,H,W = vol.shape
    vdom = np.zeros((3,D,H,W), dtype=np.float32)
    for i in range(D*H*W):
        z = i // (H*W); rem = i % (H*W); y = rem // W; x = rem % W
        M = np.array([[Jxx[z,y,x], Jxy[z,y,x], Jxz[z,y,x]],
                      [Jxy[z,y,x], Jyy[z,y,x], Jyz[z,y,x]],
                      [Jxz[z,y,x], Jyz[z,y,x], Jzz[z,y,x]]], dtype=np.float32)
        w, V = np.linalg.eigh(M)   # ascending
        v = V[:, -1]               # eigenvector for largest eigenvalue
        vdom[:, z,y,x] = v / (np.linalg.norm(v) + 1e-6)
    return vdom  # [3,D,H,W], unit vector

def build_features(patch):
    """Compute feature channels for a small 3D patch [D,H,W] -> [C,D,H,W] and names."""
    p = robust_clip_norm(patch)                            # normalize raw intensity
    # 1) raw intensity
    ch_raw = p[None, ...]
    # 2–3) LoG + DoG
    log1 = gaussian_laplace(p, sigma=1.0)
    dog = gaussian_filter(p, 1.0) - gaussian_filter(p, 2.0)
    ch_log = log1[None, ...]; ch_dog = dog[None, ...]
    # 4) gradient magnitude
    gradmag = gaussian_gradient_magnitude(p, sigma=1.0)[None, ...]
    # 5–7) Hessian eigenvalues (λ1≥λ2≥λ3)
    lam = hessian_eigs(p, sigma=1.0)                       # [3,D,H,W]
    # 8–10) Structure-tensor dominant orientation (vx,vy,vz)
    vdom = structure_tensor_orientation(p, 1.0, 2.0)       # [3,D,H,W]
    # 11) valid mask (all ones here)
    valid = np.ones_like(p, dtype=np.float32)[None, ...]
    # 12) simple curvature prior (sum |λ|)
    curv = (np.abs(lam).sum(axis=0, keepdims=True))
    # 13–15) positional coords normalized to [0,1]
    D,H,W = p.shape
    zz = np.linspace(0,1,D, dtype=np.float32)[:,None,None]
    yy = np.linspace(0,1,H, dtype=np.float32)[None,:,None]
    xx = np.linspace(0,1,W, dtype=np.float32)[None,None,:]
    Z = np.broadcast_to(zz, (D,H,W))[None,...]
    Y = np.broadcast_to(yy, (D,H,W))[None,...]
    X = np.broadcast_to(xx, (D,H,W))[None,...]
    # stack and names
    Xc = np.concatenate([ch_raw, ch_log, ch_dog, gradmag, lam, vdom, valid, curv, Z, Y, X], axis=0)
    names = [
        "raw",
        "log_sigma1", "dog_1_2",
        "gradmag_sigma1",
        "hess_lam1", "hess_lam2", "hess_lam3",
        "st_vx", "st_vy", "st_vz",
        "valid_mask",
        "curvature_sumabs",
        "pos_z", "pos_y", "pos_x",
    ]
    return Xc.astype(np.float32), names

# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser(description="Build feature channels for a small 3D patch around (z,y,x).")
    ap.add_argument("--src", required=True, help="Dir of TIFF slices or a multi-page .tif")
    ap.add_argument("--z", type=int, required=True); ap.add_argument("--y", type=int, required=True); ap.add_argument("--x", type=int, required=True)
    ap.add_argument("--depth", type=int, default=32); ap.add_argument("--height", type=int, default=160); ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--z_start", type=int, default=None, help="Optional start slice override; otherwise center on z.")
    ap.add_argument("--out", default="features_patch.npz")
    args = ap.parse_args()

    # choose the z-range to read (small window around z)
    D = args.depth
    if args.z_start is None:
        z_start = max(0, args.z - D//2)
    else:
        z_start = args.z_start

    vol = load_stack(args.src, z_start, D)                          # [D,H,W]
    Dz, Hy, Wx = vol.shape
    # center coordinates within the loaded window
    z_in = min(max(args.z - z_start, 0), Dz-1)
    patch, valid, (z0,y0,x0) = crop_center(vol, z_in, args.y, args.x, D, args.height, args.width)
    Xc, names = build_features(patch)
    np.savez_compressed(args.out, x=Xc, names=np.array(names), origin=np.array([z0+z_start, y0, x0]))
    print(f"Feature tensor: {Xc.shape} (C,D,H,W) with channels: {names}")
    print(f"Saved to: {args.out}")

if __name__ == "__main__":
    main()
