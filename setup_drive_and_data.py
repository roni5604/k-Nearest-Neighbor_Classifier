from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# Set the folder name
FOLDERNAME = 'cs231n/Assignments/assignment1/'
assert FOLDERNAME is not None, "[!] Enter the foldername"

# Append folder to sys.path
import sys
sys.path.append('/content/drive/My Drive/{}'.format(FOLDERNAME))

# Change directory and run the dataset script
%cd /content/drive/My\ Drive/$FOLDERNAME/CV7062610/
!bash get_datasets.sh
%cd /content
