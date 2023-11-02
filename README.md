# pangu-pytorch

This represents our initial attempt to replicate the Pangu-Weather model as detailed in the work by Bi et al. (2023). Our implementation closely adheres to the pseudocode provided in the GitHub repository at https://github.com/198808xc/Pangu-Weather. We have made specific adjustments to certain components, such as the PatchEmbedding and PatchRecovery layers, in alignment with the pretrained weights available in ONNX files. Notably, the attention mask is not utilized in our current implementation.

It is worth noting that ONLY LATITUDE WEIGHTED RMSE are computed and compared! ACC is not computed.

To assess the accuracy of our model, we conducted comparative tests focusing on the 24-hour horizon scores. These comparisons involve the scores reported in the original paper, the scores computed using the pretrained weights (in onnx file), and the scores generated by our reconstructed model. The tests were conducted on 2018 ERA5 data, specifically sampled at 00:00 UTC and 12:00 UTC.

<p align="left">
  <img src="fig/VIS.png" width="600" title="Figure 1. Visualization of two sample results returned by the reproduced model.">
</p>
Figure 1. Visualization of two sample results returned by the reproduced model.
  
  
  
<p align="left">
  <img src="fig/vis_onnx.png" width="600" title="Figure 2. Visualization of two sample results returned by the pretrained weights.">
</p>
Figure 2. Visualization of two sample results returned by the pretrained weights

<p align="center">
  <img src="fig/tab1.png" width="600">
</p>
<p align="center">
  <img src="fig/tab2.png" width="600">
</p>
