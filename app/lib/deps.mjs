import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
export function dependency(name){try{return require(name)}catch(error){if(!process.env.MX_RUNTIME_MODULES)throw error;return require(`${process.env.MX_RUNTIME_MODULES}/${name}`)}}
export const sharp=dependency('sharp');
export const JSZip=dependency('jszip');
