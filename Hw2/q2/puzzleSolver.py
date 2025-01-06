# Lian Natour, 207300443
# Mohammad Mhneha, 315649814

import cv2
import numpy as np
import os
import shutil
import sys

# Matches is of (3|4 X 2 X 2) size. Each row is a match - pair of (kp1, kp2) where kpi = (x, y)
def get_transform(matches, is_affine):
    # matches = matches[0]
    src_points = matches[:, 0, :]  # Extract source points (x, y)
    dst_points = matches[:, 1, :]  # Extract destination points (x, y)
    src_points = np.float32(src_points)
    dst_points = np.float32(dst_points)
    #print(src_points)
    if is_affine:
        # Ensure there are exactly 3 points for affine transformation
        if len(src_points) != 3:
            raise ValueError("Affine transformation requires exactly 3 matching points.")
        T = cv2.getAffineTransform(src_points, dst_points)
    else:
        # Ensure there are at least 4 points for homography
        if len(src_points) < 4:
            raise ValueError("Homography transformation requires at least 4 matching points.")
        T, _ = cv2.findHomography(src_points, dst_points)

    return T

def stitch(img1, img2):
    result = img1.copy()
    black_pixels_in_img1 = np.all(img1 == 0, axis=-1)
    non_black_pixels_in_img2 = np.any(img2 != 0, axis=-1)
    replacement_condition = np.logical_and(black_pixels_in_img1, non_black_pixels_in_img2)
    result[replacement_condition] = img2[replacement_condition]
    non_black_pixels_in_img1 = np.logical_not(black_pixels_in_img1)
    non_black_pixels_in_img2 = np.logical_not(np.all(img2 == 0, axis=-1))
    both_have_values = np.logical_and(non_black_pixels_in_img1, non_black_pixels_in_img2)
    result[both_have_values] = (img1[both_have_values] // 2) + (img2[both_have_values] // 2)
    return result

# Output size is (w, h)
def inverse_transform_target_image(target_img, original_transform, output_size):
    if original_transform.shape == (2, 3):
        inverse_transform = cv2.invertAffineTransform(original_transform)
        result = cv2.warpAffine(target_img, inverse_transform, output_size, flags=cv2.INTER_LINEAR)
    else:
        inverse_transform = np.linalg.inv(original_transform)
        result = cv2.warpPerspective(target_img, inverse_transform, output_size, flags=cv2.INTER_LINEAR)

    return result

# Returns list of pieces file names
def prepare_puzzle(puzzle_dir):
    edited = os.path.join(puzzle_dir, 'abs_pieces')
    if os.path.exists(edited):
        shutil.rmtree(edited)
    os.mkdir(edited)

    affine = 4 - int("affine" in puzzle_dir)

    matches_data = os.path.join(puzzle_dir, 'matches.txt')
    n_images = len(os.listdir(os.path.join(puzzle_dir, 'pieces')))

    matches = np.loadtxt(matches_data, dtype=np.int64).reshape(n_images - 1, affine, 2, 2)

    return matches, affine == 3, n_images

if __name__ == '__main__':
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'puzzles')
    lst = ['puzzle_affine_1','puzzle_affine_2','puzzle_homography_1']

    for puzzle_dir in lst:
        print(f'Starting {puzzle_dir}')

        puzzle = os.path.join(base_dir, puzzle_dir)

        pieces_pth = os.path.join(puzzle, 'pieces')
        edited = os.path.join(puzzle, 'abs_pieces')
        matches, is_affine, n_images = prepare_puzzle(puzzle)

        # Get all image files in pieces directory
        images = [image for image in os.listdir(pieces_pth) if image.lower().endswith('.jpg')]
        images.sort()  # Ensure correct order of images

        # Read the first image
        first_image_path = os.path.join(pieces_pth, images[0])
        first_image = cv2.imread(first_image_path)
        if first_image is None:
            print(f"Error: Could not read the first image {images[0]}.")
            continue

        # Save the first image to 'abs_pieces'
        first_image_output_path = os.path.join(edited, 'piece_1_relative.png')
        cv2.imwrite(first_image_output_path, first_image)

        # Initialize final puzzle image with the first image
        final_puzzle = first_image.copy()
        first_image_size = first_image.shape[1], first_image.shape[0]  # (width, height)

        # Process each subsequent piece
        for i in range(1, len(images)):
            current_image_path = os.path.join(pieces_pth, images[i])
            current_image = cv2.imread(current_image_path)
            if current_image is None:
                print(f"Error: Could not read image {images[i]}.")
                continue

            # Get the transformation matrix
            transformation_matrix = get_transform(matches[i - 1], is_affine)

            # Transform the current image to align with the first image
            transformed_image = inverse_transform_target_image(current_image, transformation_matrix, first_image_size)

            # Save the transformed image to 'abs_pieces'
            transformed_image_path = os.path.join(edited, f'piece_{i+1}_relative.png')
            cv2.imwrite(transformed_image_path, transformed_image)

            # Stitch the transformed image into the final puzzle
            final_puzzle = stitch(final_puzzle, transformed_image)

        # Save the final assembled puzzle
        sol_file = 'solution.jpg'
        solution_path = os.path.join(puzzle, sol_file)
        cv2.imwrite(solution_path, final_puzzle)
        print(f'Puzzle {puzzle_dir} solved. Solution saved as {sol_file}')
