import sys
sys.path.insert(0, 'E:/Desktop/双接口/image-fission/src')
import traceback

try:
    from v191_burn import burn_bat_logo, burn_camo_armed
    print("imports OK")
    print("burn_bat_logo OK")

    bat_src = "E:/Desktop/双接口/image-fission/jobs/smoke_v190/v190_bat_logo.jpg"
    bat_dst = "E:/Desktop/双接口/image-fission/jobs/smoke_v190/v190_bat_logo_burned_v191.jpg"
    print(f"Processing: {bat_src}")
    burn_bat_logo(bat_src, bat_dst)
    print(f"bat_logo done: {bat_dst}")

    camo_src = "E:/Desktop/双接口/image-fission/jobs/smoke_v190/v190_camo_armed.jpg"
    camo_dst = "E:/Desktop/双接口/image-fission/jobs/smoke_v190/v190_camo_armed_burned_v191.jpg"
    print(f"Processing: {camo_src}")
    burn_camo_armed(camo_src, camo_dst)
    print(f"camo_armed done: {camo_dst}")

except Exception as e:
    traceback.print_exc()
    sys.exit(1)
