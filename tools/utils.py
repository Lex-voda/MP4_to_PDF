import os
import cv2
import argparse
from fpdf import FPDF
from tqdm import tqdm
import numpy as np
from skimage.metrics import structural_similarity as ssim

def save_frames_as_pdf(frame_list, output_folder, pdf_name, overwrite=True, verbose=True):
    pdf_path = os.path.join(output_folder, pdf_name + ".pdf")
    if not overwrite and os.path.exists(pdf_path):
        print(f"PDF already exists: {pdf_path}")
        return
    
    size = (int(1920 // 3.7795), int(1080 // 3.7795))
    pdf = FPDF('P', 'mm', size)

    if verbose:
        with tqdm(total=len(frame_list), desc=f"Converting {len(frame_list)} frames to PDF", unit='frame') as pbar:
            for index, frame in enumerate(frame_list):
                # 将帧转换为图像
                image = frame

                # 将图像保存为临时文件
                temp_image_path = f"{output_folder}/temp_{index}.jpg"
                cv2.imwrite(temp_image_path, image)

                pdf.add_page()
                # 将图像添加到PDF页面中
                pdf.image(temp_image_path, 0, 0, size[0], size[1])

                os.remove(temp_image_path)

                pbar.update(1)
    else:
        for index, frame in enumerate(frame_list):
            image = frame

            # 将图像保存为临时文件
            temp_image_path = f"{output_folder}/temp_{index}.jpg"
            cv2.imwrite(temp_image_path, image)
            
            pdf.add_page()
            pdf.image(temp_image_path, 0, 0, size[0], size[1])
            
            # 删除临时文件
            os.remove(temp_image_path)
            
    pdf.output(pdf_path, "F")
    print(f"PDF created: {pdf_path}")
    
    
def similarity_compare(frame1, frame2):
    # 将帧转换为灰度图像
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 计算结构相似度
    return ssim(gray1, gray2)

def is_dominant_color(frame, threshold=2/3):
    # Convert the frame to a 1D array
    frame_1d = frame.reshape(-1, frame.shape[-1])

    # Count the frequency of each color
    unique_colors, counts = np.unique(frame_1d, axis=0, return_counts=True)

    # Calculate the frequency of each color
    frequencies = counts / frame_1d.shape[0]

    # Check if any color's frequency is above the threshold
    return any(frequency > threshold for frequency in frequencies)

def parse_args():
    parser = argparse.ArgumentParser(description='Convert videos to PDFs.')
    parser.add_argument('-if', '--input_folder', default='video', help='Path to the input video folder.')
    parser.add_argument('-of', '--output_folder', default='pdf', help='Path to the output PDF folder.')
    parser.add_argument('-st', '--similarity_threshold', type=float, default=0.95, help='Similarity threshold for frame comparison.')
    parser.add_argument('-n', '--num_workers', type=int, default=8, help='Number of worker processes.')
    parser.add_argument('-a', '--async_model', type=bool, default=False, help='Whether to async.')
    parser.add_argument('-v', '--verbose', type=bool, default=True, help='Whether to print verbose output.')
    parser.add_argument('-w', '--overwrite', type=bool, default=True, help='Whether to overwrite existing PDFs.')
    return parser.parse_args()