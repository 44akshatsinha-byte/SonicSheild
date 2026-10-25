"""Diagnostic environment check script for SonicShield."""
import sys
import shutil
import platform
import subprocess

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

def check_env():
    print("=" * 60)
    print("SONICSHIELD ENVIRONMENT DIAGNOSTICS")
    print("=" * 60)
    print(f"Platform: {platform.platform()}")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {sys.version.split()[0]}")

    packages = [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("soundfile", "soundfile"),
        ("librosa", "librosa"),
        ("praat-parselmouth", "parselmouth"),
        ("scikit-learn", "sklearn"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("streamlit", "streamlit"),
        ("plotly", "plotly"),
        ("pytest", "pytest"),
    ]

    print("\n[Package Verification]")
    all_packages_ok = True
    for display_name, import_name in packages:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [OK] {display_name:20s} (v{ver})")
        except ImportError as e:
            all_packages_ok = False
            print(f"  [FAIL] {display_name:20s} NOT INSTALLED ({e})")

    print("\n[External System Utilities]")
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        print(f"  [OK] FFmpeg found: {ffmpeg_path}")
        try:
            res = subprocess.run(["ffmpeg", "-codecs"], capture_output=True, text=True, timeout=3)
            has_amr = "amr" in res.stdout.lower() or "amr_nb" in res.stdout.lower()
            print(f"  [INFO] FFmpeg AMR-NB Support: {'Available' if has_amr else 'Unavailable'}")
        except Exception as e:
            print(f"  [WARN] Failed checking FFmpeg codecs: {e}")
    else:
        print("  [INFO] FFmpeg not found on PATH. Native ITU-T G.711 mu-law will be used.")

    print("=" * 60)
    if all_packages_ok:
        print("Status: ALL CORE DEPENDENCIES READY FOR SONICSHIELD MVP")
    else:
        print("Status: MISSING DEPENDENCIES. Please check requirements.txt.")
    print("=" * 60)

if __name__ == "__main__":
    check_env()
