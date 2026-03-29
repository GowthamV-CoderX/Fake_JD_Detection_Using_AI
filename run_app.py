import webbrowser

import uvicorn

if __name__ == "__main__":
    url = "http://127.0.0.1:8000"
    print(f"Starting Job Scam Detector website at {url}")
    webbrowser.open_new_tab(url)
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
