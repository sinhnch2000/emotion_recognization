from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import router_emotion

app = FastAPI(docs_url=None, redoc_url=None)

app.include_router(router_emotion.router)

def generate_rapidoc_html(spec_url: str) -> str:
    return """
    <!DOCTYPE html>
    <html>
    <head><title>API Docs</title></head>
    <body>
        <rapi-doc spec-url="{spec_url}"></rapi-doc>
        <script src="https://unpkg.com/rapidoc/dist/rapidoc-min.js"></script>
    </body>
    </html>
    """.format(spec_url=spec_url)


@app.get("/", response_class=HTMLResponse)
async def root():
    return generate_rapidoc_html("/openapi.json")

