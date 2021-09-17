from pathlib import Path
import numpy as np
import cv2
import pydicom
from tqdm import tqdm

DATASET = 'train'
scan_types = ['FLAIR','T1w','T1wCE','T2w']
data_root = Path("../")
out_root = Path('./voxel_preprocessed_V2')

def get_image_plane(data):
    cords = [round(j) for j in data.ImageOrientationPatient]
    return cords

def get_voxel(study_id, scan_type):
    imgs = []
    dcm_dir = data_root.joinpath(DATASET, study_id, scan_type)
    dcm_paths = sorted(dcm_dir.glob("*.dcm"), key=lambda x: int(x.stem.split("-")[-1]))
    positions = []

    for dcm_path in dcm_paths:
        img = pydicom.dcmread(str(dcm_path))
        imgs.append(img.pixel_array)
        positions.append(img.ImagePositionPatient)

    plane = get_image_plane(img)        
    voxel = np.stack(imgs)
    voxel = crop_voxel(voxel)

    rotDir = []
    rotDir.append(positions[-1][0]-positions[0][0])
    rotDir.append(positions[-1][1]-positions[0][1])
    rotDir.append(positions[-1][2]-positions[0][2])

    rotDir = np.array(rotDir)
    rotDir = rotDir / np.max(np.absolute(rotDir))
    rotDir = np.around(rotDir)

    rotVec = []

    rotVec.append(np.arctan2(rotDir[0], rotDir[1]))
    rotVec.append(np.arctan2(rotDir[1], rotDir[2]))
    rotVec.append(np.arctan2(rotDir[2], rotDir[0]))

    rotVec = np.array(rotVec)
    rotVec = rotVec / np.max(np.absolute(rotVec))
    rotVec = np.around(rotVec)

    voxel = np.rot90(voxel, plane[1], (2, 0))
    voxel = np.rot90(voxel, plane[2], (0, 1))
    voxel = np.rot90(voxel, plane[3], (1, 2))
    voxel = np.rot90(voxel, plane[5], (0, 1))

    if plane[0] == 0:
        voxel = np.flip(voxel, 1)
    if plane[4] == 0:
        voxel = np.flip(voxel, 0)
    if rotDir[1] == 1:
        voxel = np.flip(voxel, 1)
    if rotDir[2] == -1:
        voxel = np.flip(voxel, 0)

    return voxel, plane

def normalize_contrast(voxel):
    if voxel.sum() == 0:
        return voxel
    voxel = voxel - np.min(voxel)
    voxel = voxel / np.max(voxel)
    voxel = (voxel * 255).astype(np.uint8)
    return voxel

def crop_voxel(voxel):
    if voxel.sum() == 0:
        return voxel
    keep = (voxel.mean(axis=(0, 1)) > 1)
    voxel = voxel[:, :, keep]
    keep = (voxel.mean(axis=(0, 2)) > 1)
    voxel = voxel[:, keep]
    keep = (voxel.mean(axis=(1, 2)) > 1)
    voxel = voxel[keep]
    return voxel

def resize_voxel(voxel, sz=64):
    output = np.zeros((sz, sz, sz), dtype=np.uint8)

    if np.argmax(voxel.shape) == 0:
        for i, s in enumerate(np.linspace(0, voxel.shape[0] - 1, sz)):
            output[i] = cv2.resize(voxel[int(s)], (sz, sz))
    elif np.argmax(voxel.shape) == 1:
        for i, s in enumerate(np.linspace(0, voxel.shape[1] - 1, sz)):
            output[:, i] = cv2.resize(voxel[:, int(s)], (sz, sz))
    elif np.argmax(voxel.shape) == 2:
        for i, s in enumerate(np.linspace(0, voxel.shape[2] - 1, sz)):
            output[:, :, i] = cv2.resize(voxel[:, :, int(s)], (sz, sz))

    return output

for study_path in tqdm(list(data_root.joinpath(DATASET).glob("*"))):
    study_id = study_path.name
    if study_id in ['000109', '00123', '00709']:
        print("BAD ID")
        continue

    if not study_path.is_dir():
        continue

    for i, scan_type in enumerate(scan_types):
        npy_name = scan_type + '.npy'
        voxel, plane = get_voxel(study_id, scan_type)
        voxel = normalize_contrast(voxel)
#         voxel = crop_voxel(voxel)
        voxel = resize_voxel(voxel, sz=128)

        out_file = out_root / DATASET / study_id
        out_file.mkdir(exist_ok=True, parents=True)
        np.save(out_file / scan_type, voxel)