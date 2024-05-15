import os
import glob
import cv2
from natsort import natsorted
import threading
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor

from tools.ProgressBar import ProgressBar
from tools.utils import save_frames_as_pdf, similarity_compare, is_dominant_color, parse_args


class VideoToPdf:
    def __init__(self, 
                 input_folder, 
                 output_folder, 
                 similarity_threshold=0.90,
                 num_workers=10,
                 async_model=False,
                 verbose=True,
                 overwrite=True):
        self.similarity_threshold = similarity_threshold
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.num_workers = num_workers
        self.verbose = verbose
        self.overwrite = overwrite
        self.async_model = async_model
        
        self.video_name_list = [os.path.splitext(os.path.basename(filename))[0] for filename in glob.glob(os.path.join(self.input_folder, "*.mp4"))]
        self.video_name_list = natsorted(self.video_name_list)
        
        self.frame_list = []
        self.frame_count = 0
        self.fps = 0
        
        self.progressbar = ProgressBar()
        self.lock = threading.Lock()
        
        # Create output folder if not exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    
        
    def frame_vaild(self, cap, i, dominant=True):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            return False
        cap.set(cv2.CAP_PROP_POS_FRAMES, i+self.fps)
        ret, next_frame = cap.read()
        if not ret:
            return True
        similarity = similarity_compare(frame, next_frame)
        if dominant:
            return similarity > self.similarity_threshold and  not is_dominant_color(frame)
        else:
            return similarity > self.similarity_threshold
    
    def extract_frames(self, video_path, start, end):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        
        jump = self.fps
        i = start
        pre_i = i
        pre__frame_index = i
        frames = []
        flag = False
        
        ret, frame = cap.read()
        if not ret:
            return None
        frames.append(frame)
        while i + jump < end :
            next_frame_index = min(i + jump, end - 1)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame_index)
            ret, next_frame = cap.read()
            if not ret:
                break
            
            if similarity_compare(frame, next_frame) < self.similarity_threshold and self.frame_vaild(cap, next_frame_index):
                # Found a different frame
                if not flag:
                    flag = True
                    i = pre__frame_index
                    jump = self.fps
                    continue
                else:
                    flag = False
                    pre_i = i
                    i = next_frame_index
                    frames.append(next_frame)
                    frame = next_frame
                    jump = self.fps
                    with self.lock:
                        self.progressbar.update(i-pre_i, 1)
                    continue
            # No different frame found, continue to next jump
            pre__frame_index = next_frame_index
            jump *= 2
        with self.lock:
            self.progressbar.update(i-pre_i, 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, end)
        ret, last_frame = cap.read()
        cap.release()
        if not ret:
            return frames
        if similarity_compare(frames[-1], last_frame) < self.similarity_threshold:
            frames.append(last_frame)
        return frames
        
    
    def process_video(self, video_path, num_workers):
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video file")
            return
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.progressbar.total = frame_count
        self.progressbar.prefix = f"Processing {os.path.basename(video_path)}"
        self.progressbar.start()
                    
        if self.async_model:
            frames_per_worker = frame_count // num_workers
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for i in range(num_workers):
                    start = i * frames_per_worker
                    if i == num_workers - 1:  # last worker takes the remaining frames
                        end = frame_count
                    else:
                        end = start + frames_per_worker
                    futures.append(executor.submit(self.extract_frames, video_path, start, end))

                results = []
                for future in futures:
                    results.append(future.result())
            for i in range(len(results), 1):
                if similarity_compare(results[i-1][-1],results[i][0])>self.similarity_threshold:
                    results[i] = results[i][1:]
            self.frame_list = sum(results, [])
        else:
            self.frame_list = self.extract_frames(video_path, 0, frame_count)
            
        self.progressbar.clear()
        
        # Remove similar frames
        for i in range(len(self.frame_list),1):
            if similarity_compare(self.frame_list[i], self.frame_list[i-1])>self.similarity_threshold:
                self.frame_list[i] = None
        self.frame_list = [frame for frame in self.frame_list if frame is not None]
    
    
    def run(self):
        for video_name in self.video_name_list:
            video_path = os.path.join(self.input_folder, video_name+".mp4")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("Error opening video file")
                return
            
            self.process_video(video_path,self.num_workers)
            save_frames_as_pdf(self.frame_list, 
                               self.output_folder,
                               video_name, 
                               overwrite=True, 
                               verbose=True)
            self.frame_list = []
            
            
if __name__ == "__main__":
    args = parse_args()
    
    video_to_pdf = VideoToPdf(args.input_folder, 
                              args.output_folder, 
                              args.similarity_threshold, 
                              args.num_workers, 
                              args.async_model,
                              args.verbose, 
                              args.overwrite)
    video_to_pdf.run()