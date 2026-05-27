import sys

with open('/Users/summer/Large-Time-Series-Model/models/Timer.py', 'r') as f:
    content = f.read()

old = "                if self.ckpt_path.endswith('.pth'):\n                    self.backbone.load_state_dict(torch.load(self.ckpt_path))"
new = """                elif self.ckpt_path.endswith('.pth'):
                    sd = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
                    sd2 = {}
                    for k, v in sd.items():
                        clean_k = k
                        while clean_k.startswith("module."):
                            clean_k = clean_k[len("module."):]
                        while clean_k.startswith("model."):
                            clean_k = clean_k[len("model."):]
                        if not clean_k.startswith("backbone."):
                            sd2["backbone." + clean_k] = v
                        else:
                            sd2[clean_k] = v
                    self.backbone.load_state_dict(sd2, strict=True)"""

if old not in content:
    print("OLD NOT FOUND")
    idx = content.find("self.ckpt_path.endswith('.pth')")
    if idx >= 0:
        print(f"Found at idx {idx}")
        print(repr(content[idx-20:idx+200]))
    sys.exit(1)

content = content.replace(old, new, 1)
with open('/Users/summer/Large-Time-Series-Model/models/Timer.py', 'w') as f:
    f.write(content)
print("Done")
