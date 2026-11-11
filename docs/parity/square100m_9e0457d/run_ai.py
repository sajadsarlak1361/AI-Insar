import time
APP_START=time.monotonic()
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['NUMBA_NUM_THREADS']='4'
from pathlib import Path
import json,hashlib,traceback,numpy as np,rasterio
from ai_insar.safe import resolve_safe,select_measurement,read_window
from ai_insar.pipeline.s1_annotation import read_swath_from_product,read_eof_orbit
from ai_insar.pipeline.dimap_s1 import precise_orbit_table,_orbit_from_vectors
from ai_insar.pipeline.tops import burst_pixel_window
from ai_insar.pipeline import tops_ops as t
from ai_insar.pipeline.dem import ElevationModel,GeoidGrid
from ai_insar.pipeline.coherence import coherence_map
from ai_insar.pipeline.engine import Workspace,compute_interferograms
from ai_insar.pipeline.synthetic import SLCStack
from ai_insar.pipeline.multilook import multilook_array
from ai_insar.pipeline.filtering import goldstein_snap
from ai_insar.pipeline.snaphu_io import export_snaphu,run_snaphu,import_unwrapped
from ai_insar.conventions import los_from_phase
F=Path(__file__).parent;O=F/'ai-insar';O.mkdir(exist_ok=True);roi=json.loads((F/'ROI.json').read_text());bb=roi['radar_bbox'];ct=roi['coherence_context'];mask=np.load(F/'square_mask.npy');geoct=ct|{'column1':ct['column1']+50};H=bb['row1']-bb['row0'];W=bb['column1']-bb['column0'];sl=(slice(bb['row0']-ct['row0'],bb['row1']-ct['row0']),slice(bb['column0']-ct['column0'],bb['column1']-ct['column0']));D={'app':'AI Insar UNMERGED PR20 9e0457d','fresh_from_original_SAFE':True,'no_SNAP_imagery':True,'ESD':'SKIPPED','stages':{},'ROI':roi};sources=list(Path(t.__file__).parent.glob('*.py'))+[Path(__file__).parents[4] if False else Path(t.__file__)];D['source_hashes_before']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
def save():(O/'runtime.json').write_text(json.dumps(D,indent=2)+'\n')
def stage(k,fn,params=None):
 start=time.monotonic()
 try:v=fn();D['stages'][str(k)]={'status':'COMPLETE','runtime_s':time.monotonic()-start,'params':params or {}};save();print(k,'COMPLETE',D['stages'][str(k)]['runtime_s'],flush=True);return v
 except Exception as e:D['stages'][str(k)]={'status':'FAILED','runtime_s':time.monotonic()-start,'error':str(e),'traceback':traceback.format_exc()};save();raise
try:
 def split():
  products=[];mods=[]
  for date in ['20210630','20210712']:
   p=resolve_safe(next(Path('/workspace/safe-samples').glob('*'+date+'*.SAFE')));me=select_measurement(p,'IW1','VV');m=read_swath_from_product(p,me);win=burst_pixel_window(lines_per_burst=m.lines_per_burst,first=5,last=6,samples=me.samples);products.append((p,me));mods.append(m)
  delta=np.array(roi['secondary_physical_corners_and_centre'])-np.array(roi['reference_physical_corners_and_centre']);sr0=ct['row0']+int(np.floor(delta[:,0].min()))-6;sr1=ct['row1']+int(np.ceil(delta[:,0].max()))+7;sc0=ct['column0']+int(np.floor(delta[:,1].min()))-6;sc1=geoct['column1']+int(np.ceil(delta[:,1].max()))+7
  p,me=products[0];ref=read_window(p,me,6032+ct['row0'],ct['column0'],ct['row1']-ct['row0'],geoct['column1']-ct['column0']);p,me=products[1];sec=read_window(p,me,6032+sr0,sc0,sr1-sr0,sc1-sc0);np.save(O/'stage01_reference.npy',ref);np.save(O/'stage01_secondary.npy',sec);D['secondary_source_bbox']={'row0':sr0,'row1':sr1,'column0':sc0,'column1':sc1};return products,mods,ref,sec,(sr0,sc0)
 products,mods,ref,sec,ss=stage(1,split,{'swath':'IW1','pol':'VV','bursts_one_based':[5,6],'API':'burst_pixel_window + installed SAFE read_window (ROI equivalent Split, without full-burst materialization)','reference_context':geoct,'context_reason':'50 extra range samples necessary for shipped DEM-post hull; cropped after stage3, not accepted outside geographic square.'})
 def orbit():
  out=[];models=[]
  for m,fp in zip(mods,roi['orbit_files']):
   precise=read_eof_orbit(fp);records=precise_orbit_table(precise,m.line_time(6032),m.line_time(6032+3015),degree=3);orb=_orbit_from_vectors(records);models.append(t._with_orbit(m,orb));out.append(orb)
  np.save(O/'stage02_reference.npy',ref);np.save(O/'stage02_secondary.npy',sec);return models,out
 mods,orbits=stage(2,orbit,{'degree':3,'precise_EOF':roi['orbit_files'],'one_second_table':'fresh precise_orbit_table; samples unchanged'})
 with rasterio.open('/workspace/dual-app-parity/stage03-04/copernicus_roi_ellipsoid_rows.tif') as ds:
  tr=ds.transform;dem=ElevationModel(tr.f+(np.arange(ds.height)+.5)*tr.e,tr.c+(np.arange(ds.width)+.5)*tr.a,ds.read(1),-32768,'ELLIPSOIDAL_PRECONVERTED',GeoidGrid(np.array([90.,-90.]),np.array([0.,360.]),np.zeros((2,2)),'alreadyH+N EGM2008','NONE'),'same shared ellipsoidal DEM')
 early=t.RadarGrid(6032+ct['row0'],ct['column0'],ref,mods[0]);later=t.RadarGrid(6032+ss[0],ss[1],sec,mods[1]);bg=stage(3,lambda:t.backgeocode(early,later,reference_orbit=orbits[0],secondary_orbit=orbits[1],dem=dem,resampling='BISINC_5_POINT_INTERPOLATION',dem_resampling='BILINEAR_INTERPOLATION',max_solves=None,tolerance_mm=10.),{'resampling':'BISINC_5_POINT_INTERPOLATION','DEM_resampling':'BILINEAR_INTERPOLATION','dense_requested':True,'actual_geometry':'DEM-post triangulation selected by installed package','tolerance_mm':10});np.save(O/'stage03_geometry_tile_secondary.npy',bg);bg=bg[:,:ct['column1']-ct['column0']];ref=ref[:,:ct['column1']-ct['column0']];early=t.RadarGrid(early.row,early.col,ref,early.model);np.save(O/'stage03_secondary.npy',bg);np.save(O/'stage03_reference.npy',ref);D['stage3_geometry_context']=geoct;D['stage3_inside_square_nonzero']=int((abs(bg[sl])[mask]>0).sum())
 def interferogram():
  phase=t.flat_earth_phase(early,t.RadarGrid(early.row,early.col,bg,mods[1]),early_orbit=orbits[0],later_orbit=orbits[1],degree=3);topo,heights=t.topographic_residual_phase(early,t.RadarGrid(early.row,early.col,bg,mods[1]),early_orbit=orbits[0],later_orbit=orbits[1],dem=dem,degree=3);ws=Workspace(stack=SLCStack(np.stack([ref,bg]),np.array(['2021-06-30','2021-07-12']),mods[0].wavelength_m,sensor='REAL Sentinel-1'),reference_index=0);compute_interferograms(ws);x=t.subtract_phase(ws.interferograms[0],phase+topo);c=coherence_map(ref,bg,10,azimuth_window=2,product=x);np.save(O/'stage05_flat_phase.npy',phase);np.save(O/'stage05_topo_phase.npy',topo);np.save(O/'stage05_DEM_height_m.npy',heights);np.save(O/'stage05_ifg.npy',x);np.save(O/'stage05_coherence.npy',c);return x,c
 ifg,coh=stage(5,interferogram,{'subtractFlatEarthPhase':True,'requested_srpPolynomialDegree':5,'requested_srpNumberPoints':501,'limitation':'Dense as-shipped GUI geometric phase; polynomial_phase helper exists but is not wired into GUI. No synthetic polynomial sampling or code patch applied.','orbitDegree':3,'coherence_window':[2,10],'topographic_subtraction':True,'tileExtensionPercent':100,'DEM':str('/workspace/dual-app-parity/stage03-04/copernicus_roi_ellipsoid_rows.tif'),'externalDEMApplyEGM':False,'effective_coherence_window_actual':[2,10],'window_anchor':'exact forward; left=(size-1)//2, right=size//2','coherence_on':'flat-and-topo-removed IFG'})
 def deburst():
  db=t.deburst_window(ifg,mods[0],early.row,early.col);data=db.data[sl];c=coh[sl];np.save(O/'stage06_ifg.npy',data);np.save(O/'stage06_coherence.npy',c);return data,c
 db,cohbox=stage(6,deburst,{'one_burst_interior':True,'physical_bbox':bb})
 def ml():
  x=multilook_array(db,1,1,output_intensity=False);c=multilook_array(cohbox,1,1);np.save(O/'stage07_ifg.npy',x);np.save(O/'stage07_coherence.npy',c);return x,c
 ml,cm=stage(7,ml,{'range_looks':1,'azimuth_looks':1,'output_intensity':False})
 gold=stage(8,lambda:goldstein_snap(ml,alpha=1.,fft_size=64,window_size=3,adaptive=False,coherence_mask=False),{'function':'goldstein_snap','alpha':1,'fft_size':64,'step':16,'window_size':3,'adaptive':False,'zero_pad_small_crops':True});np.save(O/'stage08_ifg.npy',gold)
 stage(9,lambda:(np.save(O/'stage09_ifg.npy',gold),np.save(O/'stage09_coherence.npy',cm)),{'radar_bbox':bb,'geographic_mask_count':int(mask.sum())});lam=mods[0].wavelength_m;D['wavelength_m']=lam
 def export():
  folder=O/'snaphu';export_snaphu(folder,np.angle(gold),cm,stat_mode='DEFO',init_method='MCF',tile_rows=1,tile_columns=1,processors=1,wavelength_m=lam);common=json.loads((F/'SNAPHU_COMMON.json').read_text());p=folder/'snaphu.conf';p.write_text(p.read_text()+'\n# Same explicit physical/cost parameters as SNAP\n'+'\n'.join(k+' '+str(v) for k,v in common.items())+'\nBYTEMASKFILE square_mask.bin\n');mask.astype('uint8').tofile(folder/'square_mask.bin')
 stage(10,export,{'stat_mode':'DEFO','initialization':'MCF','tiles':[1,1],'processors':1,'common_config':str(F/'SNAPHU_COMMON.json')})
 q=stage(11,lambda:run_snaphu(O/'snaphu','/workspace/tools/snaphu-2.0.7/bin/snaphu'),{'binary':'/workspace/tools/snaphu-2.0.7/bin/snaphu'});(O/'stage11.log').write_text(q.stdout+'\n'+q.stderr)
 unw=stage(12,lambda:import_unwrapped(O/'snaphu'),{});np.save(O/'stage12_unwrapped_rad.npy',unw);los=stage(13,lambda:los_from_phase(unw,lam),{'formula':'-lambda*phase/(4*pi)','LOS_only':True,'unit':'m'});np.save(O/'stage13_displacement_m.npy',los)
 def tc():
  grid=t.RadarGrid(6032+bb['row0'],bb['column0'],np.where(mask,los,np.nan).astype('complex64'),mods[0]);height_of=lambda lat,lon:dem.ellipsoidal_height(lat,lon,method='BILINEAR_INTERPOLATION');lats,lons=t._footprint_latlon(grid,orbits[0],height_of,3);step=5/111319.49079327358;nr=max(2,int(np.ceil((max(lats)-min(lats))/step))+1);nc=max(2,int(np.ceil((max(lons)-min(lons))/step))+1);params={'out_rows':nr,'out_cols':nc,'degree':3,'resampling':'BILINEAR_INTERPOLATION','dem_resampling':'BILINEAR_INTERPOLATION','projection':'geographic'};image,bounds,heights=t.terrain_correct(grid,orbits[0],dem,**params);support_grid=t.RadarGrid(grid.row,grid.col,mask.astype('complex64'),grid.model);support,_,_=t.terrain_correct(support_grid,orbits[0],dem,**params);valid=np.isfinite(image.real)&(support.real>.999);np.save(O/'final_valid_mask.npy',valid);np.save(O/'stage14_image_m.npy',image.real);np.save(O/'stage14_heights.npy',heights);t.write_geotiff(O/'final_displacement_m.tif',np.where(valid,image.real,np.nan),bounds);D['terrain_actual_params']=params;D['final_bounds_wsen']=list(bounds);D['final_valid_count']=int(valid.sum());D['final_shape']=list(valid.shape);return np.where(valid,image.real,np.nan),bounds
 image,bounds=stage(14,tc,{'DEM_aware_bounds':True,'image_resampling':'BILINEAR_INTERPOLATION','DEM_resampling':'BILINEAR_INTERPOLATION','requested_spacing_m':5,'projection':'WGS84 geographic','no_extra_geoid':True,'API_grid_limitation':'No explicit affine/bounds parameter. Own DEM-aware extent and approximate5m angular grid; comparison reprojects to native grid.','NaN_outside_exact_polygon':True,'support_validation':'Extra actual operator projection of validity mask, included in wall time.'})
 stage(15,lambda:t.write_kmz(O/'final_displacement.kmz',image,bounds),{'API':'unmodified tops_ops.write_kmz'});D['status']='COMPLETE_THROUGH_15';D['source_hashes_after']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources};D['source_unchanged']=D['source_hashes_after']==D['source_hashes_before'];D['whole_chain_wall_s']=time.monotonic()-APP_START;D['operator_sum_s']=sum(v['runtime_s'] for v in D['stages'].values());save();print('WHOLE_CHAIN_COMPLETE',D['whole_chain_wall_s'],D['final_valid_count'],D['final_bounds_wsen'],flush=True)
except Exception as e:D['status']='FAILED';D['error']=str(e);D['traceback']=traceback.format_exc();D['whole_chain_wall_s']=time.monotonic()-APP_START;save();raise
