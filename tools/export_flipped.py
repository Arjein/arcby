"""
Fusion 360 helper (run through the Fusion MCP script runner or Scripts and Add-Ins).
Exports a body as 3MF and a second copy flipped upside-down (top face on the bed),
then opens the flipped file in Bambu Studio.

Usage inside run(): export_flipped('Base Plate', 'Base Plate')
"""
import adsk.core, adsk.fusion, zipfile, re, os, subprocess

OUT = os.path.expanduser('~/Documents/GitHub/arcby/cad/print-files')
VERTEX = re.compile(r'<vertex x="([-0-9.eE]+)" y="([-0-9.eE]+)" z="([-0-9.eE]+)"')


def export_flipped(body_name, file_stem, open_in_bambu=True):
    app = adsk.core.Application.get()
    doc = [d for d in app.documents if d.name.startswith('Robot_v1')][0]
    doc.activate()
    design = adsk.fusion.Design.cast(doc.products.itemByProductType('DesignProductType'))
    body = design.rootComponent.bRepBodies.itemByName(body_name)
    raw = os.path.join(OUT, file_stem + '.3mf')
    flipped = os.path.join(OUT, file_stem + ' - FLIPPED (print this).3mf')
    visible = body.isLightBulbOn
    body.isLightBulbOn = True            # hidden bodies export empty
    opt = design.exportManager.createC3MFExportOptions(body, raw)
    opt.meshRefinement = adsk.fusion.MeshRefinementSettings.MeshRefinementHigh
    design.exportManager.execute(opt)
    body.isLightBulbOn = visible
    zin = zipfile.ZipFile(raw)
    model = zin.read('3D/3dmodel.model').decode('utf-8')
    zmax = max(float(z) for _, _, z in VERTEX.findall(model))
    new = VERTEX.sub(lambda m: '<vertex x="%.6f" y="%.6f" z="%.6f"' % (
        float(m.group(1)), -float(m.group(2)), zmax - float(m.group(3))), model)
    zout = zipfile.ZipFile(flipped, 'w', zipfile.ZIP_DEFLATED)
    for item in zin.infolist():
        zout.writestr(item, new if item.filename == '3D/3dmodel.model' else zin.read(item.filename))
    zout.close(); zin.close()
    if open_in_bambu:
        subprocess.run(['open', '-a', '/Applications/BambuStudio.app', flipped])
    return flipped
