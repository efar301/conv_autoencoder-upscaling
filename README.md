# Convolutional Autoencoder Upscaling

This is a convolutional autoencoder designed to upscale images from **1280×720** (HD) to **1920×1080** (Full HD). The current architecture uses several convolutional layers to first downscale the input to **640×360**, then applies a series of learned upscaling operations (with a 3× scaling factor) to generate the final 1920×1080 output.

## Results Comparison

<table>
  <tr>
    <th style="text-align:center;">Low Resolution Input (1280×720)</th>
    <th style="text-align:center;">Upscaled Output (1920×1080)</th>
  </tr>
  <tr>
    <td style="text-align:center;">
      <div style="text-align:center;">
        <img src="images\comparisons_at_epoch\epoch_100_batch2_idx1_0.5.1_Huber_Loss\lowres.jpg" alt="Low Resolution Image" style="width:400px;">
        <p>Original</p>
      </div>
    </td>
    <td style="text-align:center;">
      <div style="text-align:center;">
        <img src="images\comparisons_at_epoch\epoch_100_batch2_idx1_0.5.1_Huber_Loss\upscaled.jpg" alt="Upscaled Image" style="width:400px;">
        <p>Upscaled image lost details in the wood grain</p>
      </div>
    </td>
  </tr>
  <tr>
    <td style="text-align:center;">
      <div style="text-align:center;">
        <img src="images\comparisons_at_epoch\epoch_100_batch4_idx1_0.5.1_Huber_Loss\lowres.jpg" alt="Low Resolution Image" style="width:400px;">
        <p>Original</p>
      </div>
    </td>
    <td style="text-align:center;">
      <div style="text-align:center;">
        <img src="images\comparisons_at_epoch\epoch_100_batch4_idx1_0.5.1_Huber_Loss\upscaled.jpg" alt="Upscaled Image" style="width:400px;">
        <p>Warmth added to image while upscaling</p>
      </div>
    </td>
  </tr>
</table>


---

## Current Issues

While the autoencoder shows promising results, the following issues have been observed:

- **Detail Loss in Dark Areas:**  
  Dark regions of images tend to lose fine details during upscaling.
  
- **Desaturation in Light Areas:**  
  Brighter regions often appear washed out and lack vibrancy compared to the original image.
  
- **Color Inaccuracy:**  
  Training on standard RGB images sometimes leads to colors that do not match the original exactly.
  
- **Hallucinated Sharpness:**  
  The model sometimes generates artificially sharp details. Depending on the image, this can either enhance or detract from the overall quality.
  
- **Blurred Regions:**  
  Certain areas of the output remain blurry when compared to the input, indicating room for improvement in preserving texture.

---

## Future Plans

To address these issues and further enhance the upscaling performance, the following improvements are planned:

- **Switch to YCbCr Color Space:**  
  Train the model using the YCbCr color space to potentially increase color accuracy and better preserve luminance details.
  
- **Architectural Improvements:**  
  - Experiment with progressive downscaling (reducing resolution in steps) to better preserve high-frequency information.  
  - Incorporate additional convolutional layers or residual connections to improve the quality of the reconstructed image.
  
- **Variable Resolution Scaling:**  
  Develop a model capable of handling variable upscale factors so that it is not hard coded to output 1080p images.
  
- **Real-Time Upscaling for Video and Gaming:**  
  Aim to extend the technology to upscale videos and maybe even video games.

---

