## User Defined Parameters
```
export DATADIR='path/to/pos/files/'
export RRNG='/path/to/rangefile.rrng'
export RESULTSDIR='path/to/save/output/'
export R=1.0
export O=0.5
export MINK=4
export MAXK=8
export NREPEATS=10
```

## Run Scripts

```
# clone OPTICS-APT repo if not already present
if [ ! -d 'apt' ]; then git clone https://github.com/pnnl/apt.git; fi

export PYTHONPATH=$PYTHONPATH:$PWD/apt/OPTICS-APT

export K=`seq ${MINK} ${MAXK}`
export SEED=`seq 1 ${NREPEATS}`

for S in "${DATADIR}"*.pos; do export SAMP="${S##*/}"; echo "${SAMP}"; python run_neighborhood_generation.py --datadir "${DATADIR}" --sample "${SAMP}" --rrng "${RRNG}" --savedir "${RESULTSDIR}"/neighborhoods --radius $R --overlap $O; done

for k in $K; do python run_clustering.py --datadir "${RESULTSDIR}"/neighborhoods --savedir "${RESULTSDIR}"/bulk_clustering -r $R -o $O -k $k --seeds ${SEED}; done

python run_ks_stats.py --datadir "${RESULTSDIR}"/bulk_clustering --r $R --o $O --k $K --seeds ${SEED}

python run_community_mapping.py --datadir "${RESULTSDIR}"/bulk_clustering/processed --r $R --o $O --k $K --seeds ${SEED}
```
