# -*- coding: utf-8 -*-
from flask import Flask, jsonify
from flask_cors import CORS
import traceback

# 初始化Flask实例
app = Flask(__name__, static_folder="static/dist")
# 跨域配置
CORS(app, supports_credentials=True)

# 全局状态码
CODE_SUCCESS = 200
CODE_PARAM_ERROR = 400
CODE_SERVER_ERROR = 500
CODE_NOT_FOUND = 404
CODE_METHOD_ERROR = 405


# 统一响应体
def api_response(code=CODE_SUCCESS, msg="操作成功", data=None):
    return jsonify({"code": code, "msg": msg, "data": data or {}})


# 全局异常捕获
@app.errorhandler(Exception)
def handle_all_exception(error):
    if "404" in str(error):
        return api_response(CODE_NOT_FOUND, "接口不存在")
    elif "405" in str(error):
        return api_response(CODE_METHOD_ERROR, "请求方式错误")
    else:
        app.logger.error(f"异常: {str(error)}\n{traceback.format_exc()}")
        return api_response(CODE_SERVER_ERROR, "服务器内部异常")


# 把响应函数挂载到app，供子文件调用
app.api_response = api_response

# 执行app以加载路由
from app.app import *
