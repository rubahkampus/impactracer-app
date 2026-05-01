"""Run pytest and print results."""
import subprocess, sys
result = subprocess.run(
    [sys.executable, "-m", "pytest", "--tb=short", "-q"],
    capture_output=True, text=True,
    cwd=r"C:\Users\Haidar\Documents\thesis\impactracer-app"
)
print(result.stdout)
print(result.stderr)
sys.exit(result.returncode)
