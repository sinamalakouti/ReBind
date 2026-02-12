"""
Methods required for reading/processing config files
"""

import argparse
import configparser
import os
import logging
import json

logger = logging.getLogger(__name__)

def parse_value(value_str):
    """Parse string config values into appropriate Python types."""
    # Remove quotes if present
    value_str = value_str.strip("\"'")

    # Try to parse as JSON first (handles lists, dicts, etc.)
    try:
        return json.loads(value_str)
    except json.JSONDecodeError:
        pass

    # Try to parse as boolean
    if value_str.lower() in ("true", "false"):
        return value_str.lower() == "true"

    # Try to parse as int
    try:
        return int(value_str)
    except ValueError:
        pass

    # Try to parse as float
    try:
        return float(value_str)
    except ValueError:
        pass

    # Return as string if no other type matches
    return value_str


def to_dict(parser):
    """Convert ConfigParser to dictionary with proper types."""
    confdict = {
        section: {key: parse_value(value) for key, value in parser.items(section)}
        for section in parser.sections()
    }
    return confdict


def merge_configs(base, new):
    """Merge two configurations, with new values overriding base values."""

    def convert_to_dict(cfg):
        """Convert various config types to dictionary."""
        if cfg is None:
            return {}

        if isinstance(cfg, argparse.Namespace):
            return vars(cfg)  # Convert Namespace to dict
        if isinstance(cfg, configparser.ConfigParser):
            return to_dict(cfg)
        if isinstance(cfg, dict):
            return cfg

        try:
            return dict(cfg)
        except:
            raise TypeError(f"Cannot convert type {type(cfg)} to dict: {cfg}")

    # Convert both configs to dictionaries
    base = convert_to_dict(base)
    new = convert_to_dict(new)

    if not isinstance(base, dict) or not isinstance(new, dict):
        raise TypeError(
            f"Failed to convert configs to dictionaries.\n"
            f"Got:\n"
            f"base: {type(base)}, value: {base}\n"
            f"new: {type(new)}, value: {new}"
        )

    merged = base.copy()

    for key, value in new.items():
        # Skip None values from argparse
        if value is None:
            continue

        # If both values are dicts, merge them recursively
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        # Otherwise, new value overrides base value
        else:
            merged[key] = value

    return merged


def load_config(config_path: str) -> dict:
    """Load and validate config file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    base_config_path = "./configs/base_config.ini"
    config = configparser.ConfigParser()
    config.read(config_path)

    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"Base config file not found: {base_config_path}")
    base_config = configparser.ConfigParser()
    base_config.read(base_config_path)

    config = merge_configs(base_config, config)
    log_config(config)


    return config


def log_config(cfg):
    """Log configuration in a clean, readable format."""
    logging.info("Configuration:")
    logging.info("\n" + "=" * 50)

    print("Configuration:")
    print("\n" + "=" * 50)

    def convert_section(section):
        """Convert a ConfigParser section to a dictionary with proper types."""
        if isinstance(section, (configparser.SectionProxy, configparser._SectionProxy)):
            return {k: parse_value(v) for k, v in section.items()}
        return section

    # Convert configparser to nested dict
    if isinstance(cfg, configparser.ConfigParser):
        config_dict = {}

        # Handle DEFAULT section
        if "DEFAULT" in cfg:
            config_dict["DEFAULT"] = convert_section(cfg["DEFAULT"])

        # Handle all other sections
        for section in cfg.sections():
            config_dict[section] = convert_section(cfg[section])

    else:
        config_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else cfg

    # Pretty print the config
    def format_dict(d, indent=0):
        """Format dictionary for pretty printing."""
        lines = []
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(" " * indent + f"{key}:")
                lines.extend(format_dict(value, indent + 2))
            else:
                print("instance type", type(value))
                lines.append(" " * indent + f"{key}: {value}")
        return lines

    formatted_lines = format_dict(config_dict)
    formatted_config = "\n".join(formatted_lines)

    logging.info("\nConfig:\n" + formatted_config)
    logging.info("=" * 50 + "\n")

    print("\nConfig:\n" + formatted_config)
    print("=" * 50 + "\n")
