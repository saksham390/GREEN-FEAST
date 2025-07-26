 GREEN FEAST-AI-Powered Food Waste Tracking & Penalty System to Reduce Food Wastage 
 Prepared By:TEAM NEUROVIA
 (SAKSHAM,SHIVAM CHOUBEY,LUV AGNIHOTRI)

WORKING:
1) AI Image Analysis– Identifies leftover food in thali 
2) Weight Estimation – Calculates wasted quantity (grams).   
3) Auto-Penalty – Charges guests via UPI if waste exceeds limit.   
4) Social Impact–Donates fines to charity. 

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
