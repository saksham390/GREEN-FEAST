 GREEN FEAST-AI-Powered Food Waste Tracking & Penalty System to Reduce Food 
Wastage  
 AI Image Analysis– Identifies leftover food in thali 
 Weight Estimation – Calculates wasted quantity (grams).   
 Auto-Penalty – Charges guests via UPI if waste exceeds limit.   
 Social Impact–Donates fines to charity. 

    A[Camera Input] --> B{Face Detection}
    B -->|Detected| C[Person Identification]
    B -->|Not Detected| D[Continue Monitoring]
    C --> E[Initial Plate Scan]
    E --> F[Food Quantification]
    F --> G[Monitor Consumption]
    G --> H{Significant Waste?}
    H -->|Yes| I[Send SMS Alert]
    H -->|No| G
    I --> J[Generate Payment QR]
