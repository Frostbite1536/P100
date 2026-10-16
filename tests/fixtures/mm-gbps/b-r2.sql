-- Hand-written synthetic perf fixture, not hardware measurements.
CREATE TABLE test_backend_ops (op_name TEXT, op_params TEXT, supported INTEGER, time_us REAL, test_mode TEXT);
INSERT INTO test_backend_ops VALUES ('MUL_MAT', 'type_a=q4_K,type_b=f32,m=4096,n=8,k=4096,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0', 1, 60.0, 'perf');
INSERT INTO test_backend_ops VALUES ('MUL_MAT', 'type_a=q4_K,type_b=f32,m=4096,n=16,k=4096,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0', 1, 120.0, 'perf');
INSERT INTO test_backend_ops VALUES ('MUL_MAT', 'type_a=q8_0,type_b=f32,m=1024,n=8,k=4096,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0', 1, 60.0, 'perf');
INSERT INTO test_backend_ops VALUES ('MUL_MAT', 'type_a=q8_0,type_b=f32,m=1024,n=16,k=4096,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0', 1, 120.0, 'perf');
