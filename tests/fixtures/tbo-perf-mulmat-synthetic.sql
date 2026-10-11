-- SYNTHETIC: hand-written parsing/arithmetic fixture; no hardware measurements.
CREATE TABLE IF NOT EXISTS test_backend_ops (
  backend_name TEXT, build_commit TEXT, op_name TEXT, op_params TEXT,
  test_mode TEXT, supported INTEGER, passed INTEGER, time_us REAL, n_runs INTEGER
);
INSERT INTO test_backend_ops VALUES ('CUDA0','synthetic','MUL_MAT','type_a=q8_0,type_b=f32,m=32,n=1,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0','perf',1,1,2,100);
INSERT INTO test_backend_ops VALUES ('CUDA0','synthetic','MUL_MAT','type_a=q4_1,type_b=f32,m=32,n=1,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0','perf',1,1,2,200);
INSERT INTO test_backend_ops VALUES ('CUDA0','synthetic','MUL_MAT','type_a=q4_0,type_b=f32,m=32,n=1,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0','perf',1,1,5,50);
INSERT INTO test_backend_ops VALUES ('CUDA0','synthetic','MUL_MAT','type_a=q8_0,type_b=f32,m=32,n=4,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0','perf',1,1,10,10);
INSERT INTO test_backend_ops VALUES ('CUDA0','synthetic','MUL_MAT','type_a=q8_0,type_b=f16,m=32,n=1,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3]','perf',1,1,999,100);
INSERT INTO test_backend_ops VALUES ('CUDA0','synthetic','MUL_MAT','type_a=f16,type_b=f32,m=32,n=1,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3]','perf',1,1,999,100);
INSERT INTO test_backend_ops VALUES ('CUDA0','synthetic','MUL_MAT','type_a=q8_0,type_b=f32,m=32,n=9,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3]','perf',1,1,999,100);
INSERT INTO test_backend_ops VALUES ('CUDA0','synthetic','MUL_MAT','type_a=q8_0,type_b=f32,m=32,n=2,k=256,bs=[1,1],nr=[1,1],per=[0,2,1,3]','perf',1,1,999,100);
INSERT INTO test_backend_ops VALUES ('CUDA0','synthetic','MUL_MAT','type_a=q8_0,type_b=f32,m=32,n=2,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=512','perf',1,1,999,100);
INSERT INTO test_backend_ops VALUES ('CUDA0','synthetic','MUL_MAT','type_a=q8_0,type_b=f32,m=32,n=2,k=256,bs=[2,1],nr=[1,1],per=[0,1,2,3]','perf',1,1,999,100);
INSERT INTO test_backend_ops VALUES ('CUDA0','synthetic','MUL_MAT','type_a=q8_0,type_b=f32,m=32,n=2,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],o=2','perf',1,1,999,100);
INSERT INTO test_backend_ops VALUES ('CUDA0','synthetic','MUL_MAT','type_a=q8_0,type_b=f32,m=32,n=2,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],m_v=64,pad=4','perf',1,1,999,100);
INSERT INTO test_backend_ops VALUES ('CUDA0','synthetic','MUL_MAT','type_a=q8_0,type_b=f32,m=32,n=2,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3]','perf',0,0,0,0);
INSERT INTO test_backend_ops VALUES ('CUDA0','synthetic','MUL_MAT','type_a=q8_0,type_b=f32,m=32,n=2,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3]','test',1,1,999,100);
INSERT INTO test_backend_ops VALUES ('CPU','synthetic','MUL_MAT','type_a=q8_0,type_b=f32,m=32,n=1,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3]','perf',1,1,999,100);
