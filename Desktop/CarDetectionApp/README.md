# 🚗 Car Detection (YOLOv8) - Streamlit (Hugging Face Spaces)

This Space uses a YOLOv8 model (`best.pt`) to detect cars in uploaded images.

## Files
- `app.py` - Streamlit app
- `best.pt` - trained weights
- `requirements.txt` - dependencies

## How to run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Hugging Face Spaces (recommended)
1. Create a new Space
2. SDK: **Streamlit**
3. Hardware: **GPU**
4. Commit/push: `app.py`, `best.pt`, `requirements.txt`, `README.md`

The Space will start automatically and show the Streamlit UI.

