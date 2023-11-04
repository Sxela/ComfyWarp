#(c) Alex Spirin 2023

import hashlib, os, sys, glob, subprocess, pathlib


def generate_file_hash(input_file):
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
        raise Exception(f'Frame source for {outdir_prefix} not found at {source_path}\nPlease specify an existing source path.')
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
            raise Exception(f'Couldn`t extract frames from {source_path}\nPlease specify an existing source path.')
      elif os.path.isdir(source_path):
        self.frame_paths = glob.glob(os.path.join(source_path, '*.*'))
        if len(self.frame_paths)<1:
          raise Exception(f'Found 0 frames in {source_path}\nPlease specify an existing source path.')
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

    else: raise Exception(f'Frame source for {outdir_prefix} not found at {source_path}\nPlease specify an existing source path.')
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
        raise Exception(f'Frame source not found at {source_path}\nPlease specify an existing source path.')
    if os.path.exists(source_path):
      if os.path.isfile(source_path):
        raise Exception(f'{source_path} is a file. Please specify path to a folder.')
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
  if max_dim>max_size:
    ratio = max_size/max_dim
    new_size = ((int(x*ratio)//divisible_by*divisible_by),(int(y*ratio)//divisible_by*divisible_by))
    return new_size
  return size