"""CDR (Conflict Detection & Resolution) package."""

__version__ = "0.1.0"
__author__ = "Somnath"
__email__ = "somnathab3@gmail.com"

from .geodesy import haversine_nm, bearing_rad, cpa_nm

__all__ = ["haversine_nm", "bearing_rad", "cpa_nm"]
