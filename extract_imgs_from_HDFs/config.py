import os

class Config:
    """Wrapper class that enables attribute-style access to config keys."""
    def __init__(self, config):
        self.config = config

    def __getattr__(self, name):
        return self.config[name]


def get_config(path):
    """
    Load configuration from file.
    If the config file does not exist, use default region settings.
    """
    config = {}

    if not os.path.exists(path):
        # Use default region (Mid Area)
        config['start_col'] = 504
        config['end_col'] = 840
        config['start_row'] = 176
        config['end_row'] = 405
        config['window_row'] = 500
        config['window_col'] = 800
    else:
        # Read from configuration file
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()

                # Remove inline comments starting with '#'
                comment_idx = line.find('#')
                if comment_idx != -1:
                    line = line[:comment_idx]

                # Parse key:value pairs
                parts = line.split(':')
                if len(parts) != 2:
                    continue
                key, value = parts
                config[key] = int(value)

    # Compute padded area to match the window aspect ratio
    fill_row, fill_col = get_fill_size(
        config['end_row'] - config['start_row'],
        config['end_col'] - config['start_col'],
        config['window_row'],
        config['window_col']
    )

    config['fill_row'] = fill_row
    config['fill_col'] = fill_col

    return Config(config)


def get_fill_size(area_row, area_col, win_row, win_col):
    """
    Compute the padded annotation area size so that it matches
    the aspect ratio of the display window.

    area_row, area_col: size of the labeled region
    win_row, win_col:   target window size for display
    """
    win_ratio = win_row / win_col
    area_ratio = area_row / area_col

    # If annotation region is vertically more stretched than the window,
    # expand columns based on row count.
    if area_ratio > win_ratio:
        fill_row = area_row
        fill_col = fill_row / win_ratio
        # Ensure fill_col is integer
        while abs(fill_col - int(fill_col)) > 1e-8:
            fill_row += 1
            fill_col = fill_row / win_ratio
    else:
        fill_col = area_col
        fill_row = fill_col * win_ratio
        # Ensure fill_row is integer
        while abs(fill_row - int(fill_row)) > 1e-8:
            fill_col += 1
            fill_row = fill_col * win_ratio

    return int(fill_row), int(fill_col)
