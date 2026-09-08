import sys
sys.path.insert(0, 'E:/Desktop/双接口/image-fission/src')
from v192_burn import burn_bat_logo, burn_camo_armed

bat_src = "E:/Desktop/双接口/image-fission/jobs/smoke_v190/v190_bat_logo.jpg"
bat_dst = "E:/Desktop/双接口/image-fission/jobs/smoke_v190/v190_bat_logo_burned_v192.jpg"
burn_bat_logo(bat_src, bat_dst)

camo_src = "E:/Desktop/双接口/image-fission/jobs/smoke_v190/v190_camo_armed.jpg"
camo_dst = "E:/Desktop/双接口/image-fission/jobs/smoke_v190/v190_camo_armed_burned_v192.jpg"
burn_camo_armed(camo_src, camo_dst)
print("done")
