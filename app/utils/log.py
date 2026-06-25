"""结构化日志封装"""
import logging
import structlog


def get_logger(name: str):
    """获取一个具名 logger。供 service 与节点使用。"""
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    return structlog.get_logger(name)
