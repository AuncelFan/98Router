from flask import send_from_directory
from flask import request
from app import app
import io

from app.router import Router


# 主页面
@app.route("/", methods=["GET"])
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/assets/<path:filename>")
def assets(filename):
    return send_from_directory(app.static_folder + "/assets", filename)


# POST 解析kml文件，返回解析结果
@app.route("/api/parse_kml", methods=["POST"])
def parse_kml():
    try:
        kml_file = request.files.get("kml_file")
        if not kml_file:
            return app.api_response(code=400, msg="缺少kml_file参数")
        bytes_io = io.BytesIO()
        kml_file.save(bytes_io)
        bytes_io.seek(0)
        router = Router.from_kml(bytes_io)
        metrics = router.get_metrics()
        return app.api_response(data=metrics)
    except Exception as e:
        app.logger.error(f"解析KML文件异常: {str(e)}")
        return app.api_response(code=500, msg="解析KML文件异常")
