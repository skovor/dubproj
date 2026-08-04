#!/usr/bin/env python3
import json
import shutil
from pathlib import Path

ROOT = Path(r"C:\Users\juand\Desktop\moddeutsch\p3r_text_pipeline")
REBUILT_DIR = ROOT / "awb_rebuilt"
UE_DIR = Path(r"C:\Users\juand\Desktop\Reloaded-II\Mods\p3r.doblaje.aleman\UnrealEssentials")
INDEX = json.loads((ROOT / "awb_index.json").read_text(encoding="utf-8"))

def main():
    copied = 0
    for bank_name, info in INDEX.items():
        full_bank = f"Voice_Event_{bank_name}" if not bank_name.startswith("Voice_Event_") else bank_name
        acb_src = REBUILT_DIR / full_bank / f"{full_bank}.acb"
        awb_src = REBUILT_DIR / full_bank / f"{full_bank}.awb"
        if not acb_src.exists() or not awb_src.exists():
            continue

        rel_internal = Path(info["internal"])
        # Obtener ruta relativa dentro de Stream/
        str_path = str(rel_internal).replace("\\", "/")
        if "CriData/Stream/en/" in str_path:
            sub = str_path.split("CriData/Stream/en/")[-1]
        elif "CriData/Stream/" in str_path:
            sub = str_path.split("CriData/Stream/")[-1]
        else:
            sub = rel_internal.name

        parent_rel = Path(sub).parent

        for lang_folder in ["", "de", "en"]:
            if lang_folder:
                dst_dir = UE_DIR / "P3R/Content/Xrd777/CriData/Stream" / lang_folder / parent_rel
            else:
                dst_dir = UE_DIR / "P3R/Content/Xrd777/CriData/Stream" / parent_rel

            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(acb_src, dst_dir / acb_src.name)
            shutil.copy2(awb_src, dst_dir / awb_src.name)
            copied += 1

    print(f"¡Copiados {copied} pares ACB+AWB emparejados a UnrealEssentials!")

if __name__ == "__main__":
    main()
