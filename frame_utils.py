#(c) Alex Spirin 2023

import hashlib, os, sys, glob, subprocess, pathlib, platform
import zipfile
from shutil import copy
import requests

def generate_file_hash(input_file):
    """Generates has for video vile based on name, size, creation time
    """
    # Get file name and metadata
    file_name = os.path.basename(input_file)
    file_size = os.path.getsize(input_file)
    creation_time = os.path.getctime(input_file)

    # Generate hash
    hasher = hashlib.sha256()
    hasher.update(file_name.encode('utf-8'))
    hasher.update(str(file_size).encode('utf-8'))
    hasher.update(str(creation_time).encode('utf-8'))
    file_hash = hasher.hexdigest()

    return file_hash

def createPath(filepath):
    os.makedirs(filepath, exist_ok=True)

def extractFrames(video_path, output_path, nth_frame, start_frame, end_frame):
  createPath(output_path)
  print(f"Exporting Video Frames (1 every {nth_frame})...")
  try:
    for f in [o.replace('\\','/') for o in glob.glob(output_path+'/*.jpg')]:
      pathlib.Path(f).unlink()
  except:
    print('error deleting frame ', f)
  vf = f'select=between(n\\,{start_frame}\\,{end_frame}) , select=not(mod(n\\,{nth_frame}))'
  if os.path.exists(video_path):
    try:
        subprocess.run(['ffmpeg', '-i', f'{video_path}', '-vf', f'{vf}', '-vsync', 'vfr', '-q:v', '2', '-loglevel', 'error', '-stats', f'{output_path}/%06d.jpg'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    except:
        subprocess.run(['ffmpeg.exe', '-i', f'{video_path}', '-vf', f'{vf}', '-vsync', 'vfr', '-q:v', '2', '-loglevel', 'error', '-stats', f'{output_path}/%06d.jpg'], stdout=subprocess.PIPE).stdout.decode('utf-8')

  else:
    sys.exit(f'\nERROR!\n\nVideo not found: {video_path}.\nPlease check your video path.\n')


class FrameDataset():
  def __init__(self, source_path, outdir_prefix='', videoframes_root='', update_on_getitem=False):
    self.frame_paths = None
    image_extenstions = ['jpeg', 'jpg', 'png', 'tiff', 'bmp', 'webp']
    self.update_on_getitem = update_on_getitem
    self.source_path = source_path
    if not os.path.exists(source_path):
      if len(glob.glob(source_path))>0:
        self.frame_paths = sorted(glob.glob(source_path))
      else:
        raise FileNotFoundError(f'Frame source for {outdir_prefix} not found at {source_path}\nPlease specify an existing source path.')
    if os.path.exists(source_path):
      if os.path.isfile(source_path):
        if os.path.splitext(source_path)[1][1:].lower() in image_extenstions:
          self.frame_paths = [source_path]
        hash = generate_file_hash(source_path)[:10]
        out_path = os.path.join(videoframes_root, outdir_prefix+'_'+hash)

        extractFrames(source_path, out_path,
                        nth_frame=1, start_frame=0, end_frame=999999999)
        self.frame_paths = glob.glob(os.path.join(out_path, '*.*'))
        self.source_path = out_path
        if len(self.frame_paths)<1:
            raise FileNotFoundError(f'Couldn`t extract frames from {source_path}\nPlease specify an existing source path.')
      elif os.path.isdir(source_path):
        self.frame_paths = glob.glob(os.path.join(source_path, '*.*'))
        if len(self.frame_paths)<1:
          raise FileNotFoundError(f'Found 0 frames in {source_path}\nPlease specify an existing source path.')
    extensions = []
    if self.frame_paths is not None:
      for f in self.frame_paths:
            ext = os.path.splitext(f)[1][1:]
            if ext not in image_extenstions:
              raise Exception(f'Found non-image file extension: {ext} in {source_path}. Please provide a folder with image files of the same extension, or specify a glob pattern.')
            if not ext in extensions:
              extensions+=[ext]
            if len(extensions)>1:
              raise Exception(f'Found multiple file extensions: {extensions} in {source_path}. Please provide a folder with image files of the same extension, or specify a glob pattern.')

      self.frame_paths = sorted(self.frame_paths)

    else: raise FileNotFoundError(f'Frame source for {outdir_prefix} not found at {source_path}\nPlease specify an existing source path.')
    print(f'Found {len(self.frame_paths)} frames at {source_path}')

  def __getitem__(self, idx):
    if self.update_on_getitem:
       self.frame_paths = glob.glob(os.path.join(self.source_path, '*.*'))

    idx = min(idx, len(self.frame_paths)-1)
    return self.frame_paths[idx]

  def __len__(self):
    return len(self.frame_paths)
  

class StylizedFrameDataset():
  def __init__(self, source_path):
    self.frame_paths = None
    self.source_path = source_path
    if not os.path.exists(source_path):
        raise FileNotFoundError(f'Frame source not found at {source_path}\nPlease specify an existing source path.')
    if os.path.exists(source_path):
      if os.path.isfile(source_path):
        raise NotADirectoryError(f'{source_path} is a file. Please specify path to a folder.')
      elif os.path.isdir(source_path):
        self.frame_paths = glob.glob(os.path.join(source_path, '*.*'))

  def __getitem__(self, idx):
    self.frame_paths = glob.glob(os.path.join(self.source_path, '*.*'))
    if len(self.frame_paths)==0:
      return ''

    idx = min(idx, len(self.frame_paths)-1)
    return self.frame_paths[idx]

  def __len__(self):
    return len(self.frame_paths)
  

def get_size(size, max_size, divisible_by=8):
  divisible_by = int(divisible_by)
  x,y = size 
  max_dim = max(size)
  # if max_dim>max_size:
  ratio = max_size/max_dim
  new_size = ((int(x*ratio)//divisible_by*divisible_by),(int(y*ratio)//divisible_by*divisible_by))
  return new_size
  # return size

def find_ffmpeg(start_dir):
  files = glob.glob(f"{start_dir}/**/*.*", recursive=True)
  for f in files: 
    if platform.system() == 'Linux':
      if f.endswith('ffmpeg'): return f
    elif f.endswith('ffmpeg.exe'): return f
  return None

def download_ffmpeg(start_dir):
  ffmpeg_url = 'https://github.com/GyanD/codexffmpeg/releases/download/6.0/ffmpeg-6.0-full_build.zip'
  print('ffmpeg.exe not found, downloading...')
  r = requests.get(ffmpeg_url, allow_redirects=True, timeout=60)
  print('downloaded, extracting')
  open('ffmpeg-6.0-full_build.zip', 'wb').write(r.content)
 
  with zipfile.ZipFile('ffmpeg-6.0-full_build.zip', 'r') as zip_ref:
        zip_ref.extractall(f'{start_dir}/')

  copy(f'{start_dir}/ffmpeg-6.0-full_build/bin/ffmpeg.exe', f'{start_dir}/')
  return f'{start_dir}/ffmpeg.exe'


def get_ffmpeg():
  start_dir = os.getcwd()
  ffmpeg_path = find_ffmpeg(start_dir)
  if ffmpeg_path is None or not os.path.exists(ffmpeg_path):
    ffmpeg_path = download_ffmpeg(start_dir)
  return ffmpeg_path  

def save_video(indir, video_out, batch_name='', start_frame=1, last_frame=-1, fps=30, output_format='h264_mp4', use_deflicker=False):
  ffmpeg_path = get_ffmpeg()
  print('Found ffmpeg at: ', ffmpeg_path)
  os.makedirs(video_out, exist_ok=True)
  indir = indir.replace('\\','/')
  image_path = f"{indir}/{batch_name}_%06d.png"

  postfix = ''
  if use_deflicker:
      postfix+='_dfl'

  indir_stem = indir.replace('\\','/').split('/')[-1]
  out_filepath = f"{video_out}/{indir_stem}_{postfix}.{output_format.split('_')[-1]}"
  if last_frame == -1:
    last_frame = len(glob.glob(f"{indir}/*.png"))

  cmd = [ffmpeg_path,
        '-y',
        '-vcodec',
        'png',
        '-r',
        str(fps),
        '-start_number',
        str(start_frame),
        '-i',
        image_path,
        '-frames:v',
        str(last_frame+1),
        '-c:v']

  if output_format == 'h264_mp4':
    cmd+=['libx264',
        '-pix_fmt',
        'yuv420p']
  elif output_format == 'qtrle_mov':
    cmd+=['qtrle',
      '-vf',
      f'fps={fps}']
  elif output_format == 'prores_mov':
    cmd+=['prores_aw',
      '-profile:v',
      '2',
      '-pix_fmt',
      'yuv422p10',
      '-vf',
      f'fps={fps}']
    
  if use_deflicker:
    cmd+=['-vf','deflicker=mode=pm:size=10']
  cmd+=[out_filepath]

  process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  stdout, stderr = process.communicate()
  if process.returncode != 0:
      print(stderr)
      raise RuntimeError(stderr)
  else:
      print(f"The video is ready and saved to {out_filepath}")
      return 0
