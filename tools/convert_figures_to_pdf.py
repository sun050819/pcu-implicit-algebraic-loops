"""
Convert PNG figures to vector PDF format for IEEE TEVC.
This script embeds PNG figures into PDF containers (for true vector output,
re-run the original plotting scripts with savefig format='pdf').
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

fig_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "ieee_tevc_paper", "figures")

png_files = [f for f in os.listdir(fig_dir) if f.endswith('.png')]

for png_file in png_files:
    png_path = os.path.join(fig_dir, png_file)
    pdf_path = os.path.join(fig_dir, png_file.replace('.png', '.pdf'))
    
    img = mpimg.imread(png_path)
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w/100, h/100), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.axis('off')
    fig.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Converted: {png_file} -> {png_file.replace('.png', '.pdf')}")

print("\nAll figures converted to PDF.")
print("Note: For true vector graphics, re-run the original plotting scripts with:")
print("  plt.savefig('figure.pdf', format='pdf', dpi=300)")
