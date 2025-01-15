import matplotlib.pyplot as plt
from PIL import Image


def visualize(query_text, pred_texts, query_image_path, pred_image_paths):
    """Visualize the query and predicted images and texts.

    Args:
        query_text (_type_): _description_
        pred_texts (_type_): _description_
        query_image_path (_type_): _description_
        pred_image_paths (_type_): _description_
    """
    query_image = Image.open(query_image_path)
    pred_images = [Image.open(pred) for pred in pred_image_paths]

    # Determine the number of rows needed
    num_images = len(pred_images) + 1
    num_cols = 5
    num_rows = (num_images + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    # Display query image and text
    axes[0].imshow(query_image)
    axes[0].set_title(f"Query: {query_text}", wrap=True)
    axes[0].axis("off")

    # Display predicted images and texts
    for i, (pred_image, pred_text) in enumerate(zip(pred_images, pred_texts)):
        # Wrap long text
        wrapped_text = "\n".join(
            pred_text[i : i + 30] for i in range(0, len(pred_text), 30)
        )
        axes[i + 1].imshow(pred_image)
        axes[i + 1].set_title(f"Pred: {wrapped_text}", wrap=True)
        axes[i + 1].axis("off")

    # Hide any remaining axes
    for j in range(num_images, len(axes)):
        axes[j].axis("off")

    plt.tight_layout(pad=1.0)  # Increase padding between plots
    plt.subplots_adjust(hspace=0, wspace=1)  # Increase space between query and preds

    # Increase space between elements in the same row
    for ax in axes:
        ax.margins(x=0.1, y=0.1)

    plt.show()
