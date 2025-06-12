import cv2
import os
import re


def generate_video_from_images(folder_path, output_path, framerate=30):
    # get image files
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    if not image_files:
        print(f"skip: no image {folder_path}")
        return

    # sort images based on the number in the filename
    images = sorted(image_files, key=lambda f: int(
        re.search(r'\d+', f).group()))

    # get image size
    first_image_path = os.path.join(folder_path, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    # create video file
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(
        *'mp4v'), framerate, (width, height))

    for img_name in images:
        img_path = os.path.join(folder_path, img_name)
        frame = cv2.imread(img_path)
        out.write(frame)

    out.release()
    print(f"Created: {output_path}")


def main(base_dir='diffnpm', framerate=30):
    for subdir in sorted(os.listdir(base_dir)):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path) and subdir.startswith("iter"):
            output_path = f"{subdir_path}.mp4"
            generate_video_from_images(subdir_path, output_path, framerate)


if __name__ == '__main__':
    main()
