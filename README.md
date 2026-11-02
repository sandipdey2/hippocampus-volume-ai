# Hippocampus Volume Quantification for Alzheimer’s Progression

End-to-end AI system that automatically measures hippocampal volume from T2-weighted MRI and delivers a DICOM report into a clinical PACS/viewer workflow.

## Overview

Hippocampal atrophy is a key biomarker of Alzheimer’s disease. This project implements the full pipeline:

1. **Dataset curation** – clean and explore the Medical Decathlon Hippocampus dataset  
2. **U-Net segmentation** – train a 2D U-Net (Mean Dice 0.9004, Mean Jaccard 0.8202)  
3. **Clinical integration** – DICOM inference → Secondary Capture report → Orthanc PACS → OHIF viewer

## Repository Structuresection1/          # Dataset curation & EDA
out/             # Cleaned NIfTI pairs + EDA notebook
section2/          # Model training
out/             # results.json, TensorBoard screenshots
section3/          # Clinical deployment
out/             # inference_dcm.py, report.dcm, validation plan
Hippocampus_Complete_Project_Report.pdf
Clinician_Explanation_Email.txttext## Results

| Metric          | Value   |
|-----------------|---------|
| Mean Dice       | 0.9004  |
| Mean Jaccard    | 0.8202  |

## How to Run

### Section 1 – EDA
```bash
cd section1
jupyter notebook Final_Project_EDA.ipynbSection 2 – TrainingBashcd section2/src
python run_ml_pipeline.pySection 3 – InferenceBash# Start Orthanc and OHIF first
cd section3/src
python inference_dcm.py /data/TestVolumesLarge Filesmodel.pth (~119 MB) and the full NIfTI dataset exceed GitHub’s file size limits and are not included in this repository.

Download the trained model separately and place it in section2/src/ or section3/src/ before running inference.ValidationA 1–2 page clinical validation plan is included in section3/out/validation_plan.txt.LicenseEducational project for the Udacity AI for Healthcare Nanodegree.
