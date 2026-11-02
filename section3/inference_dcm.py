"""
HippoVolume.AI - DICOM inference and report generation
"""
import os
import sys
import datetime
import time
import shutil
import subprocess

import numpy as np
import pydicom

from PIL import Image, ImageFont, ImageDraw

from inference.UNetInferenceAgent import UNetInferenceAgent


def load_dicom_volume_as_numpy_from_list(dcmlist):
    if not dcmlist:
        raise ValueError("Empty DICOM list")
    slices = [np.flip(dcm.pixel_array).T for dcm in sorted(dcmlist, key=lambda dcm: getattr(dcm, "InstanceNumber", 0))]
    hdr = dcmlist[0]
    hdr.PixelData = None
    return np.stack(slices, 2), hdr


def get_predicted_volumes(pred):
    voxel_volume = 1.0  # mm³
    volume_ant = float(np.sum(pred == 1) * voxel_volume)
    volume_post = float(np.sum(pred == 2) * voxel_volume)
    total_volume = volume_ant + volume_post
    return {"anterior": volume_ant, "posterior": volume_post, "total": total_volume}


def create_report(inference, header, orig_vol, pred_vol):
    pimg = Image.new("RGB", (1000, 1000), color=(0, 0, 0))
    draw = ImageDraw.Draw(pimg)

    try:
        header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=36)
        main_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=20)
    except Exception:
        header_font = ImageFont.load_default()
        main_font = ImageFont.load_default()

    draw.text((20, 20), "HippoVolume.AI Report", fill=(255, 255, 255), font=header_font)

    report_text = (
        f"Patient ID: {getattr(header, 'PatientID', 'N/A')}\n"
        f"Study Date: {getattr(header, 'StudyDate', 'N/A')}\n"
        f"Series Description: {getattr(header, 'SeriesDescription', 'N/A')}\n\n"
        f"Anterior Hippocampus Volume: {inference['anterior']:.1f} mm3\n"
        f"Posterior Hippocampus Volume: {inference['posterior']:.1f} mm3\n"
        f"Total Hippocampus Volume: {inference['total']:.1f} mm3\n\n"
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    draw.multiline_text((20, 100), report_text, fill=(255, 255, 255), font=main_font)
    return pimg


def save_report_as_dcm(header, report_img, path):
    img_array = np.array(report_img)
    ds = pydicom.Dataset()
    ds.PatientID = getattr(header, "PatientID", "UNKNOWN")
    ds.PatientName = getattr(header, "PatientName", "UNKNOWN")
    ds.StudyInstanceUID = getattr(header, "StudyInstanceUID", pydicom.uid.generate_uid())
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    ds.Modality = "OT"
    ds.SeriesDescription = "HippoVolume.AI Report"
    ds.InstanceNumber = "1"
    ds.Rows, ds.Columns = img_array.shape[0], img_array.shape[1]
    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "RGB"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = img_array.tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    ds.save_as(path)
    print(f"Report saved to {path}")


def get_series_for_inference(study_dir):
    """Return list of DICOM datasets from the first suitable series."""
    series_for_inference = []
    for root, dirs, files in os.walk(study_dir):
        dcms = []
        for f in files:
            if f.lower().endswith(".dcm"):
                try:
                    dcms.append(pydicom.dcmread(os.path.join(root, f)))
                except Exception:
                    continue
        if 10 < len(dcms) < 200:
            return dcms
        if dcms and not series_for_inference:
            series_for_inference = dcms
    return series_for_inference


def os_command(command):
    print("Running:", command)
    sp = subprocess.Popen(["/bin/bash", "-c", command])
    sp.communicate()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference_dcm.py <folder_containing_studies>")
        sys.exit(1)

    studies_dir = sys.argv[1]
    if not os.path.isdir(studies_dir):
        print(f"Directory not found: {studies_dir}")
        sys.exit(1)

    # Find study sub-folders
    study_dirs = [os.path.join(studies_dir, d) for d in os.listdir(studies_dir)
                  if os.path.isdir(os.path.join(studies_dir, d))]
    if not study_dirs:
        print("No study sub-folders found in", studies_dir)
        sys.exit(1)

    study_dir = sorted(study_dirs, key=lambda d: os.stat(d).st_mtime, reverse=True)[0]
    print(f"Using study: {study_dir}")

    series = get_series_for_inference(study_dir)
    if not series:
        print("No DICOM series found")
        sys.exit(1)

    volume, header = load_dicom_volume_as_numpy_from_list(series)
    print(f"Volume shape: {volume.shape}")

    print("Running inference...")
    model_path = os.path.join(os.path.dirname(__file__), "model.pth")
    if not os.path.exists(model_path):
        model_path = "/home/workspace/section3/src/model.pth"
    inference_agent = UNetInferenceAgent(device="cpu", parameter_file_path=model_path)

    pred_label = inference_agent.single_volume_inference_unpadded(np.array(volume))
    pred_volumes = get_predicted_volumes(pred_label)
    print(f"Volumes: {pred_volumes}")

    report_save_path = "/home/workspace/section3/out/report.dcm"
    os.makedirs(os.path.dirname(report_save_path), exist_ok=True)
    report_img = create_report(pred_volumes, header, volume, pred_label)
    save_report_as_dcm(header, report_img, report_save_path)

    # Push to Orthanc
    os_command(f"storescu 127.0.0.1 4242 {report_save_path} -v")

    print("Done.")
    print(f"Total hippocampal volume: {pred_volumes['total']:.1f} mm3")
