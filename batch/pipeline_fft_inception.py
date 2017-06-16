ds = Dataset(index=DatasetIndex(index = files), batch_class=EcgBatch)

ds.cv_split(0.8)
pp = (ds.train.p.load(src)
              .add_ref(DIR_ECG + 'REFERENCE.csv')
              .drop_noise()
              .augment_fs([('delta', {'loc': 250}),
                           ('delta', {'loc': 350}),
                           ('none', {})])
              .segment(3000, 3000, pad=True)
              .train_fft_inception(nb_epoch=1, batch_size=1000))

ppt = (ds.test.p.load(src)
              .add_ref(DIR_ECG + 'REFERENCE.csv')
              .drop_noise()
              .augment_fs([('delta', {'loc': 250}),
                           ('delta', {'loc': 350}),
                           ('none', {})])
              .segment(3000, 3000, pad=True)
              .calc_loss())

ppr = (ds.p.train_report())

NUM_ITER = 2
for i in range(NUM_ITER):
    b1 = pp.next_batch(batch_size=len(ds.train.index), shuffle=True, n_epochs=1, prefetch=0) 
    b2 = ppt.next_batch(batch_size=len(ds.test.index), shuffle=True, n_epochs=1, prefetch=0) 
    b3 = ppr.next_batch(batch_size=len(ds.index), shuffle=False, n_epochs=1, prefetch=0) 