2. Background and Related Work 

Applied to the Vesuvius problem, this structure enables the model to route information through a learned graph reflecting the hierarchy of physical phenomena—e.g., from papyrus geometry → fiber texture → surface continuity → ink probability—mirroring the true dependencies in the data. More generally, this graph-structured expert network offers a foundation for hierarchical reasoning in AI: scalable systems that learn both what to compute and how to coordinate computation dynamically across multiple specialized modules. 

3. Model Architecture 

3.3.4 Input 

Given the  

- input as tokens from token stem 

- graph model in center 

- output layer 

3.4 Expert Learning 

3.5 Outputs 

4. Training 

Data is formatted as 3D µCT scans of Vesuvius scrolls 

Scanning capabilities vary in resolution and energy levels for each scroll 

Scan conditions are given for each scroll 

Give model raw signal, enhanced features, iteration feedback 

4.1 Feature Selection 

Intensity: Raw µCT intensity is used as the ground-truth sensor data, containing subtle density contrasts between papyrus and possible ink. (1 channel) 

Enhancement Filters: Local contrast, Laplacian-of-Gaussians, and difference-of-Gaussians are used here to enhance faint boundaries or faint strokes. (3 channels) 

Hessian Eigenvalues: Hessian eigenvalues describe local curvature or intensity of surfaces, helping to separate smooth papyrus fibers from sharp cracks or ink strokes. Each eigenvalue uses its own channel. (3 channels 

Structure-Tensor Outputs: Encodes the dominant fiber orientation of papyrus in each voxel neighborhood. This helps the model distinguish background fiber lines from true ink strokes. (3 channels) 

Sheet normal estimate: A coarse estimate of the local papyrus surface normal. Crucial for geometry-aware inference. (1 channel) 

Valid-voxel mask: A binary channel marking whether the voxel belongs to material or background air. (1 channel) 

Geometric Difficulty Prior: Derived from a lightweight surface tracer. Reduces confusion in heavily compressed or damaged spots. (1 channel) 

Positional encodings: Sinusoidal features of (coordinates. Help the net generalize across tile positions and retain physical scaling. (3 channels) 

4.2 Input Stem 

 

Figure 1. Input stage architecture 

Raw input is first processed and formatted as a voxel tensor, 
𝑥∈ℝ[𝐵, 𝐶, 𝐷, 𝐻, 𝑊]
x
∈
ℝ
B
,
 
C
,
 
D
,
 
H
,
 
W
 
, where 
𝐵 
B
 
 
is batch size, 
𝐶 
C
 
 
is the number of channels for each voxel, and 
𝐷  
, 
𝐻  
, and 
𝑊  
are the depth, height, and width dimensions of each voxel, respectively. The tensor is then passed through a 3D convolutional stem consisting of: 

Conv3D(C → 64, kernel=3, stride=1, padding=1) 

Projects the multi-channel feature stack into a 64-dimensional latent space while preserving spatial resolution. 

GroupNorm 

Normalizes channel activations, stabilizing training across diverse scan intensities and feature scales. 

SiLU activation 

Introduces non-linearity and smooth gradients for better convergence. 

Repeat block (1-3) 

A second identical block further refines local voxel-level structure and prepares features for downsampling in the encoding stage. 

4.3 Encoder 

 

Figure 2. Encoding stage architecture 

After the input stage has fused the raw feature stack into a unified latent representation 
[𝐵, 64, 𝐷, 𝐻, 𝑊]
B
,
 
64
,
 
D
,
 
H
,
 
W
 
, the model must learn broader spatial context—not just local voxel neighborhoods. The encoder stage achieves this by progressively downsampling while increasing channel capacity, enabling each feature to capture larger papyrus structures such as folds, cracks, and ink strokes spanning millimeters. The architecture is as follows: 

First Downsampling Layer 

A convolutional block Conv3D(64 → 96, kernel=3, stride=2, padding=1) halves the spatial resolution in each dimension while expanding the channel depth. This effectively doubles the receptive field, allowing each encoded feature to “see” a larger physical region of papyrus while preserving local detail. 

Residual Blocks 

At this reduced resolution, the network applies residual blocks of the form: 

𝑦=𝑥+𝐹(𝑥)
y
=
x
+
F
x
 
 

Where 
𝐹(𝑥)=𝐶𝑜𝑛𝑣3𝐷 → 𝐺𝑟𝑜𝑢𝑝𝑁𝑜𝑟𝑚 → 𝑆𝑖𝐿𝑢 → 𝐶𝑜𝑛𝑣3𝐷 → 𝐺𝑟𝑜𝑢𝑝𝑁𝑜𝑟𝑚
F
x
=
C
o
n
v
3
D
 
→
 
G
r
o
u
p
N
o
r
m
 
→
 
S
i
L
u
 
→
 
C
o
n
v
3
D
 
→
 
G
r
o
u
p
N
o
r
m
 
. Skip connections stabilize training and ensure that finer-grained information is preserved while deeper features learn curvature, fiber orientation, and multi-voxel continuity. Adding lightweight channel attention (e.g., Squeeze-and-Excite) at the end of each block helps emphasize rare but informative features such as thin ink strokes. 

Second Downsampling Layer 

Another stride-2 convolution, Conv3D(96 → 128, kernel=3, stride=2, padding=1), further reduces spatial resolution. This stage extracts mid-to-large scale context, critical for recognizing papyrus topology that cannot be inferred from local patches alone. 

Residual Refinement 

Residual blocks at this scale refine global structure while suppressing high-frequency noise, balancing detail retention with robustness against imaging artifacts. 

Output 

The encoder produces a multi-scale feature pyramid with high resolution features preserve preserving fine ink-like textures and mid and low resolution features capturing broader papyrus geometry and damage patterns. These features are later passed, via skip connections, into the Recursive Decoding Stage to enable both local detail recovery and global structural reasoning. 

The model should be able to understand broader context, not just these voxels, so the encoder stage down samples the output from the input stage. Output is fed into Conv3D(64 → 96, kernel=3, stride=2, padding=1), which makes it so each figure looks at a bigger piece of the papyrus (larger receptive field). After down sampling is the residual blocks (Conv3D → GroupNorm → SiLU → Conv3D → GroupNorm) + skip connection. Another identical layer should be added to further increase the receptive field size. 

4.4 Post-Processing and Verification 

The outputs of the graph decoding stage flow into task heads representing geometry (surface charts) and ink probability maps, but these remain in model space. To produce interpretable results, the system requires a final post-processing stage that converts predictions into usable 2D scroll facsimiles while ensuring reliability through verification. 

Flattening and unwrapping 

Surface charts predicted by the geometry head are mapped into 2D planes using an as-rigid-as-possible (ARAP) or Laplacian warping algorithm. This ensures that papyrus curvature is unfolded with minimal distortion, preserving relative distances between glyphs. 

Provenance Mapping 

Every 2D pixel inherits a voxel provenance record linking it back to the original 3D coordinates in the CT volume. This mapping guarantees that each stroke seen in 2D can be verified against its 3D origin, enabling scholars to trace ambiguous regions back to raw data. 

Verification and quality assurance 

Glyphness scorer: a lightweight CNN/Transformer trained on ink fragments evaluates whether predicted strokes resemble authentic ink glyphs rather than papyrus fibers or noise. 

Consistency Check: evaluates whether adjacent glyphs form plausible sequences based on paleographic priors.  

Feedback hook 

Regions with low glyphness, inconsistent provenance, or high uncertainty are flagged for reprocessing. These flagged patches are routed back into the Recursive Decoding Stage, enabling another refinement pass. This recursive loop continues until uncertainty drops below a confidence threshold. 

Scroll reconstruction 

Once confidence is adequate, flattened surfaces and ink overlays are stitched together into larger contiguous sheets. Overlapping regions are aligned to ensure continuity of text across surface charts. The final output is a reconstructed, legible facsimile of the scroll, ready for scholarly interpretation. 

 

 

5. Architecture 

Experts should be able to discover their roles without manual labels 

Sparse MoE with learned router conditioned on geometry/uncertainty (curvature, fiber direction, ink confidence) 

Add load balancing + entropy terms so traffic spreads 

Capacity limits/expert dropout so no single expert hoovers everything 

A diversity/specialization loss 

Trained in EM-like loop 

E step assigns tiles to top-k experts by current loss/uncertainty 

M step updates the experts on their assigned tiles 

To bootstrap purpose, each expert should be given a weak prior via expert-specific augmentations (e.g., ultra-curled patches, torn seams, glossy artifacts) and tiny auxiliary heads so gradients nudge them toward distinct competencies 

Main ink/segmentation loss should be shared 

Trained on competence-aware curriculum that increasingly routes ‘hard’ patches to experts that historically reduced loss for similar patches 

Monitor with specialization metrics (routing entropy per expert, mutual information between expert ID and patch attributes) and guard against collapse with stochastic routing noise and periodic router resets.  

Net effect: experts self-organize into “curl-repair,” “tear-inpainting,” “fiber-aligned denoising,” “thin-stroke ink” specialists—learned, not hand-assigned 

 

 

 

 

Figure 2. Encoding stage architecture  

First Downsampling Layer  

A convolutional block Conv3D(64 → 96, kernel=3, stride=2, padding=1) halves the spatial resolution in each dimension while expanding the channel depth. This effectively doubles the receptive field, allowing each encoded feature to “see” a larger physical region of papyrus while preserving local detail.  

Residual Blocks  

At this reduced resolution, the network applies residual blocks of the form:  

𝑦=𝑥+𝐹(𝑥) y = x + F x 

Where 𝐹(𝑥)=𝐶𝑜𝑛𝑣3𝐷 → 𝐺𝑟𝑜𝑢𝑝𝑁𝑜𝑟𝑚 → 𝑆𝑖𝐿𝑢 → 𝐶𝑜𝑛𝑣3𝐷 → 𝐺𝑟𝑜𝑢𝑝𝑁𝑜𝑟𝑚 F x = C o n v 3 D   →   G r o u p N o r m   →   S i L u   →   C o n v 3 D   →   G r o u p N o r m 

. Skip connections stabilize training and ensure that finer-grained information is preserved while deeper features learn curvature, fiber orientation, and multi-voxel continuity. Adding lightweight channel attention (e.g., Squeeze-and-Excite) at the end of each block helps emphasize rare but informative features such as thin ink strokes.  

Second Downsampling Layer  

Another stride-2 convolution, Conv3D(96 → 128, kernel=3, stride=2, padding=1), further reduces spatial resolution. This stage extracts mid-to-large scale context, critical for recognizing papyrus topology that cannot be inferred from local patches alone.  

Residual Refinement  

Residual blocks at this scale refine global structure while suppressing high-frequency noise, balancing detail retention with robustness against imaging artifacts.  

Output  

The encoder produces a multi-scale feature pyramid with high resolution features preserve preserving fine ink-like textures and mid and low resolution features capturing broader papyrus geometry and damage patterns. These features are later passed, via skip connections, into the Recursive Decoding Stage to enable both local detail recovery and global structural reasoning.  

The model should be able to understand broader context, not just these voxels, so the encoder stage down samples the output from the input stage. Output is fed into Conv3D(64 → 96, kernel=3, stride=2, padding=1), which makes it so each figure looks at a bigger piece of the papyrus (larger receptive field). After down sampling is the residual blocks (Conv3D → GroupNorm → SiLU → Conv3D → GroupNorm) + skip connection. Another identical layer should be added to further increase the receptive field size. 

 

 

 

 

 

 

 

5.3 Decoding Stage 

The encoding stage produces a multi-scale representation that captures both local papyrus texture and broader geometric context. However, not all regions of the scroll present the same level of difficulty: some patches contain clear ink strokes on smooth surfaces, while others are heavily curled, cracked, or indistinguishable from fibers. Recursive decoding introduces adaptivity and specialization. 

Router 

The router examines encoded features along with derived statistics such as curvature, sheet thickness, estimated fiber orientation, and ink uncertainty. Based on this analysis, it assigns each patch to one of two paths: 

Direct Path: high confidence patches are passed directly to the final task heads 

Expert Path: low confidence or ambiguous patches are routed to specialized refinement experts 

The router used a Mixture-of-Experts (MoE) gating system with Load balancing to ensure experts receive roughly equal traffic, entropy regularization to avoid degenerate routing, and dropout to prevent collapse onto a single expert. 

Expert Modules 

Experts are lightweight and specialized networks designed to refine problematic regions. Each expert is encouraged through weak priors, augmentations, and auxiliary losses to specialize in distinct failure modes, such as curl or crease repair, tear healing, fiber-aligned denoising, thin-stroke ink enhancement, and noise suppression. After processing experts return refined features back to the router. 

Recursion and Iteration 

The controller reassess uncertainty after expert refinement. If confidence remains low, the same patch can be routed through additional experts or even revisited by the same expert under new conditions. This loop allows progressive refinement—the system can try again until sufficient confidence is achieved rather than failing on the first pass. 

Task Heads 

Once confidence is adequate, features flow into the two main decoding heads. The geometry head uses an SDF-based predictor reconstructing papyrus surface charts. The ink head is a depth-invariant UNet decoder producing ink probability maps and uncertainty estimates. These outputs server as the foundation for later post-processing and validation. 

5.4 Post-Processing and Verification 

The outputs of the recursive decoding stage represent geometry (surface charts) and ink probability maps, but these remain in model space. To produce interpretable results, the system requires a final post-processing stage that converts predictions into usable 2D scroll facsimiles while ensuring reliability through verification. 

Flattening and unwrapping 

Surface charts predicted by the geometry head are mapped into 2D planes using an as-rigid-as-possible (ARAP) or Laplacian warping algorithm. This ensures that papyrus curvature is unfolded with minimal distortion, preserving relative distances between glyphs. 

Provenance Mapping 

Every 2D pixel inherits a voxel provenance record linking it back to the original 3D coordinates in the CT volume. This mapping guarantees that each stroke seen in 2D can be verified against its 3D origin, enabling scholars to trace ambiguous regions back to raw data. 

Verification and quality assurance 

Glyphness scorer: a lightweight CNN/Transformer trained on ink fragments evaluates whether predicted strokes resemble authentic ink glyphs rather than papyrus fibers or noise. 

Consistency Check: evaluates whether adjacent glyphs form plausible sequences based on paleographic priors.  

Feedback hook 

Regions with low glyphness, inconsistent provenance, or high uncertainty are flagged for reprocessing. These flagged patches are routed back into the Recursive Decoding Stage, enabling another refinement pass. This recursive loop continues until uncertainty drops below a confidence threshold. 

Scroll reconstruction 

Once confidence is adequate, flattened surfaces and ink overlays are stitched together into larger contiguous sheets. Overlapping regions are aligned to ensure continuity of text across surface charts. The final output is a reconstructed, legible facsimile of the scroll, ready for scholarly interpretation. 