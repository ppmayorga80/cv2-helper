import cv2
from cv2_helper import imshow, PreloadImages, wait_key

if __name__ == '__main__':
    img = PreloadImages.COLOR.CAT_01
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Launch the window
    win = imshow(
        window_title="Pixel Inspector",
        image=img,
        # rgb_text_font_size=16,
        # show_grid=True,
        # show_rgb_text=False,
        # grid_tone=120,  #x RGB tuple or hex string like "#CCCCCC"
        # show_grid_when_px_gt=10,  # Show grid when 1 pixel > 10 screen px
        # show_rgb_text_when_px_gt=50  # Show R/G/B values when 1 pixel > 80 screen px
    )
    wait_key(win)
