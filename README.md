# AI-Assisted Product Identification and Warranty Claim Management

## Overview
This repository contains the full implementation of an AI-assisted 
warranty claim management system developed for ELAJO Elektriska AB. 
The system automates product identification from field photographs 
and integrates the result into a digital warranty claim workflow.

## System Components
- **Product identification pipeline:** Barcode detection (ZXing) + 
  DINOv2-Large embedding retrieval via FAISS + CLIP ViT-L/14 re-ranking
- **Backend:** Python FastAPI connected to Microsoft SQL Server
- **Frontend:** React single-page application
- **Augmentation:** Index-time photorealistic augmentation using Albumentations

## Key Features
- Automatic barcode decoding for instant product identification
- Zero-shot visual similarity search across 1,501 industrial products
- Order-filtered search mode achieving above 95% identification accuracy
- Automatic ALEM09 warranty eligibility checking
- Pre-filled digital warranty claim form with PDF generation

## Models Used
- DINOv2-Large (facebook/dinov2-large)
- CLIP ViT-L/14 (openai/clip-vit-large-patch14)
  
