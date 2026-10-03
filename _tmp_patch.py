p = "networks/lora_anima/network.py"
s = open(p, encoding="utf-8").read()
old = "if effective_module_class is DoRALoRAModule and cfg.dora_detach_norm:"
new = """if (
                    effective_module_class in (DoRALoRAModule, DoKrLoRAModule)
                    and cfg.dora_detach_norm
                ):"""
assert old in s
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("network ok")

p = "configs/gui-methods/lokr.toml"
s = open(p, encoding="utf-8").read()
old = """use_lokr = true
lokr_factor = -1

# 可选叠加开关（对应其它训练器的 Lora Dora / Lora Rs 勾选框）：
# use_dora=true → DoKr（LoKr 方向 + 逐通道幅度，LyCORIS dokr 语义）；
# rs_lora=true  → 有效缩放 alpha/sqrt(r)。两个可以同时开。
use_dora = false
rs_lora = false"""
new = """use_lokr = true
lokr_factor = -1

# 默认 DoKr（LoKr 方向 + 可训练幅度）——detach 范数使速度与普通 LoRA 持平。
# 想练纯 LoKr：把 use_dora 改回 false。rs_lora=true → 有效缩放 alpha/sqrt(r)。
use_dora = true
dora_detach_norm = true
rs_lora = false"""
assert old in s
s = s.replace(old, new, 1)
s = s.replace('label = "LoKr"', 'label = "DoKr"')
s = s.replace(
    'description = "LoRA + Kronecker product — more expressive per parameter; loads in-repo only."',
    "description = \"DoKr: Kronecker direction + trainable magnitude (detach norm) — expressive AND fast; loads in-repo only.\"",
)
s = s.replace('output_name = "anima_lokr"', 'output_name = "anima_dokr"')
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("lokr.toml ok")
