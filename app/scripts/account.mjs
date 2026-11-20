import {readFile,mkdir} from 'node:fs/promises';
import path from 'node:path';
import {hashPassword,writeJSON} from '../lib/core.mjs';
let text='';for await(const chunk of process.stdin)text+=chunk;
const input=JSON.parse(text.replace(/^\uFEFF/,''));if(typeof input.username!=='string'||!/^[a-zA-Z0-9_.-]{3,50}$/.test(input.username))throw Error('账号需为 3–50 位字母、数字、下划线、点或短横线');
const root=process.env.MX_ROOT||'D:\\图片AI';await mkdir(path.join(root,'data'),{recursive:true});await writeJSON(path.join(root,'data','owner.json'),{username:input.username,...hashPassword(input.password)});console.log('账号已设置。');
