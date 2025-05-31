# setup_logger.py
import sys
import re
import logging
from colorama import Fore, Style, init
from typing import Optional

class ServerStyleFormatter(logging.Formatter):
  def format(self, record):
    msg = record.getMessage()
    record_name = record.name[:22].ljust(22)  # Truncate/pad to 22 chars
    logger_name = f"{Fore.MAGENTA}{record_name}{Style.RESET_ALL}"
    level_field = f"{record.levelname}:".ljust(8)

    # Format URLs in bold
    url_pattern = r'https?://[^\s]+'
    msg = re.sub(url_pattern, lambda m: f"{Fore.CYAN}{m.group(0)}{Style.RESET_ALL}", msg)

    # Format HTTP status codes
    msg = re.sub(r'200 OK', f"{Fore.GREEN}200 OK{Style.RESET_ALL}", msg)
    msg = re.sub(r'404 Not Found', f"{Fore.RED}404 Not Found{Style.RESET_ALL}", msg)

    # Variable Detection Framework
    # Pattern 1: Numbers (likely variables)
    msg = re.sub(r'\b(\d+)\b', f"{Fore.CYAN}\\1{Style.RESET_ALL}", msg)

    # Pattern 2: Single/double quoted strings (likely string variables)
    msg = re.sub(r"'([^']*)'", f"'{Fore.CYAN}\\1{Style.RESET_ALL}'", msg)
    msg = re.sub(r'"([^"]*)"', f'"{Fore.CYAN}\\1{Style.RESET_ALL}"', msg)

    # Pattern 3: Dictionary-style patterns and common variable indicators
    msg = re.sub(r'\[([^\]]+)\]', f"[{Fore.CYAN}\\1{Style.RESET_ALL}]", msg)

    # Pattern 4: Parenthetical expressions (often function results or calculations)
    msg = re.sub(r'\((\d+[^)]*)\)', f"({Fore.CYAN}\\1{Style.RESET_ALL})", msg)

    # Pattern 5: File extensions and technical identifiers
    msg = re.sub(r'\.(\w{2,4})\b', f".{Fore.CYAN}\\1{Style.RESET_ALL}", msg)

    # Level-based formatting
    if record.levelno == logging.INFO:
      log_level = f"{Fore.GREEN}{level_field}{Style.RESET_ALL}"
    elif record.levelno == logging.ERROR:
      log_level = f"{Fore.RED}{level_field}{Style.RESET_ALL}"
    elif record.levelno == logging.WARNING:
      log_level = f"{Fore.YELLOW}{level_field}{Style.RESET_ALL}"
    elif record.levelno == logging.DEBUG:
      log_level = f"{Fore.BLUE}{level_field}{Style.RESET_ALL}"
    else:
      log_level = f"{Fore.GREEN}{level_field}{Style.RESET_ALL}"

    return f"{logger_name} {log_level} {msg}"

_logger_configured = False

def initialize_logger(
        package_name: str,
        is_development: bool = False,
        log_level: str = "INFO",
        log_format: Optional[str] = None) -> logging.Logger:
  """Initialize logger with environment-adaptive formatting"""
  # Prevent multiple initialization
  global _logger_configured
  if _logger_configured:
    return logging.getLogger(package_name)
  _logger_configured = True

  if is_development:
    init(autoreset=True)
    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = ServerStyleFormatter()
    handler.setFormatter(formatter)
    logging.basicConfig(
      level=getattr(logging, log_level),
      handlers=[handler]
    )
  else:
    default_format = log_format or "%(asctime)s %(name)s %(levelname)s %(message)s"
    logging.basicConfig(
      level=getattr(logging, log_level),
      format=default_format,
      stream=sys.stdout,
      force=True
    )

  return logging.getLogger(package_name)
