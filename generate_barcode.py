from barcode import Code128
from barcode.writer import ImageWriter

value = "0335000001554444122115"

barcode = Code128(
    value,
    writer=ImageWriter()
)

filename = barcode.save(
    "barcode_ok",
    {
        "module_width": 0.4,   # bar thickness (important)
        "module_height": 50,   # bar height
        "quiet_zone": 10,      # REQUIRED for zbar
        "font_size": 12,
        "text_distance": 5,
        "dpi": 300
    }
)

print("Generated:", filename)
