# # For the pneumonia image (label 1)
# visualize_gradcam("https://raw.githubusercontent.com/HayBeeCoder/AIMS_Medical_Imaging/refs/heads/main/pneumonia-image-label-1.png", model, device)

# # For the normal image (label 0)
# visualize_gradcam("https://raw.githubusercontent.com/HayBeeCoder/AIMS_Medical_Imaging/refs/heads/main/pneumonia-image-label-0.png", model, device)

def visualize_gradcam(image_url, model, device, cmap='jet', alpha=0.5):
    """
    Visualize Grad-CAM for all convolutional layers of a model given an image URL.
    
    Parameters:
    - image_url (str): URL of the image to analyze
    - model (nn.Module): PyTorch model to analyze
    - device (torch.device): Device to run the model on
    - cmap (str): Colormap for the Grad-CAM overlay (default: 'jet')
    - alpha (float): Transparency of the overlay (default: 0.5)
    """
    import requests
    from io import BytesIO
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image, ImageOps
    from torchvision import transforms
    
    model.eval()
    
    # Load your image from a URL
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert('L')  # Convert to grayscale
    image = ImageOps.exif_transpose(image)
    
    transform_human = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # For model input: resize, convert to tensor, ensure 1 channel
    transform_model = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Use same normalization as training
    ])
    image_human = transform_human(image)
    image_model = transform_model(image)
    
    # Add batch dimension for model
    input_image = image_model.unsqueeze(0).to(device)
    
    # Forward pass
    output = model(input_image)  # logits
    pred_class = output.argmax(dim=1)  # predicted class
    score = output[0, pred_class]  # largest logit
    
    print(f"Predicted class index: {pred_class.item()}")
    
    # Grad-CAM code
    # Get all Conv2d layers from model.features
    layers = [m for m in model.features if isinstance(m, nn.Conv2d)]
    
    # Storage
    layer_activations = {}
    layer_gradients = {}
    
    def forward_hook(module, inp, out):
        layer_activations[module] = out.detach()
        def backward_hook(grad):
            layer_gradients[module] = grad.detach()
        out.register_hook(backward_hook)
    
    # Register hooks
    handles = [layer.register_forward_hook(forward_hook) for layer in layers]
    
    # Forward + backward
    model.zero_grad(set_to_none=True)
    output = model(input_image)
    score = output[0, pred_class]
    score.backward()
    
    # Remove forward hooks
    for h in handles:
        h.remove()
    
    # Compute and visualize Grad-CAM for each layer
    fig, axs = plt.subplots(1, len(layers), figsize=(15, 5))
    for i, layer in enumerate(layers):
        acts = layer_activations[layer]  # [1, C, H, W]
        grads = layer_gradients[layer]  # [1, C, H, W]
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        grad_cam = (weights * acts).sum(dim=1)  # [1, H, W]
        grad_cam = torch.nn.functional.interpolate(
            grad_cam.unsqueeze(0),
            size=input_image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        grad_cam = grad_cam.relu().cpu().detach().numpy()
        axs[i].imshow(image_human.squeeze(), cmap='gray')
        axs[i].imshow(grad_cam, alpha=alpha, cmap=cmap)
        axs[i].set_title(f"Grad-CAM layer {i+1}")
        axs[i].axis("off")
    plt.tight_layout()
    plt.show()
    
    return fig