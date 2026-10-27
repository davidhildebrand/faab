"""Stdlib only: list the sheets of an .xlsx and, for one sheet, its size and whether it has notes, merges,
conditional formats or data validation (to know what formatting has to survive)."""
import re
import sys
import zipfile

z = zipfile.ZipFile(sys.argv[1])
wb = z.read("xl/workbook.xml").decode("utf-8")
rels = z.read("xl/_rels/workbook.xml.rels").decode("utf-8")
target = dict(re.findall(r'Id="([^"]+)"[^>]*Target="([^"]+)"', rels))
target.update({k: v for v, k in re.findall(r'Target="([^"]+)"[^>]*Id="([^"]+)"', rels)})
for name, rid in re.findall(r'<sheet [^>]*name="([^"]+)"[^>]*r:id="([^"]+)"', wb):
    path = "xl/" + target.get(rid, "?").lstrip("/").replace("xl/", "")
    x = z.read(path).decode("utf-8") if path in z.namelist() else ""
    dim = re.search(r'<dimension ref="([^"]+)"', x)
    print(f"{name:35s} {path:28s} {dim.group(1) if dim else '?':10s} merges {x.count('<mergeCell ')}, "
          f"cond.formats {x.count('<conditionalFormatting')}, validations {x.count('<dataValidation ')}, "
          f"hyperlinks {x.count('<hyperlink ')}")
print("comments/notes files:", [n for n in z.namelist() if "comment" in n.lower()])
