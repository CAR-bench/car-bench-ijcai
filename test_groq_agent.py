import subprocess
import sys

print("=" * 80)
print("?? GROQ AGENT TEST")
print("=" * 80)

result = subprocess.run(
    [sys.executable, "src/track_1_agent_under_test/hf_test.py"],
    capture_output=True,
    text=True
)

print(result.stdout)
if result.stderr:
    print("Errors:", result.stderr)

if result.returncode == 0:
    print("? Test passed!")
else:
    print("? Test failed!")

print("\n" + "=" * 80)
print("? All tests completed!")
print("=" * 80)
