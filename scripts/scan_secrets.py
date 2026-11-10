import re
import subprocess
import sys

def main():
    res = subprocess.run(["git", "log", "-p", "--all"], capture_output=True, text=True, errors="replace")
    content = res.stdout

    patterns = [
        re.compile(r"gsk_[a-zA-Z0-9]{20,}"),
        re.compile(r"sk-[a-zA-Z0-9]{20,}"),
        re.compile(r"AIza[0-9A-Za-z-_]{35}"),
        re.compile(r"ghp_[a-zA-Z0-9]{36}"),
        re.compile(r"github_pat_[a-zA-Z0-9_]{50,}"),
        re.compile(r"(?:GROQ|OPENAI|MODEL)_API_KEY\s*[:=]\s*[\"']?[a-zA-Z0-9_\-]{20,}"),
    ]

    found = []
    for line in content.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            for pat in patterns:
                m = pat.search(line)
                if m:
                    # check if mock / fake
                    if "mock" in line.lower() or "example" in line.lower() or "placeholder" in line.lower():
                        continue
                    found.append((line.strip(), m.group(0)))

    print(f"Scanned {len(content.splitlines())} lines of full git history.")
    if found:
        print(f"ALERT: Found {len(found)} potential secret matches:")
        for line, match in found:
            print(f"  {line}")
        sys.exit(1)
    else:
        print("PASS: Zero real API keys or secrets detected in full git history.")

if __name__ == "__main__":
    main()

