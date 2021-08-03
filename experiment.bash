python run.py dataset.src_task=Ar dataset.tgt_task=Cl
python run.py dataset.src_task=Ar dataset.tgt_task=Pr
python run.py dataset.src_task=Ar dataset.tgt_task=Rw

python run.py dataset.src_task=Cl dataset.tgt_task=Ar
python run.py dataset.src_task=Cl dataset.tgt_task=Pr
python run.py dataset.src_task=Cl dataset.tgt_task=Rw

python run.py dataset.src_task=Pr dataset.tgt_task=Ar
python run.py dataset.src_task=Pr dataset.tgt_task=Cl
python run.py dataset.src_task=Pr dataset.tgt_task=Rw

python run.py dataset.src_task=Rw dataset.tgt_task=Ar
python run.py dataset.src_task=Rw dataset.tgt_task=Cl
python run.py dataset.src_task=Rw dataset.tgt_task=Pr

python run.py config=dann dataset.src_task=Ar dataset.tgt_task=Cl
python run.py config=dann dataset.src_task=Ar dataset.tgt_task=Pr
python run.py config=dann dataset.src_task=Ar dataset.tgt_task=Rw

python run.py config=dann dataset.src_task=Cl dataset.tgt_task=Ar
python run.py config=dann dataset.src_task=Cl dataset.tgt_task=Pr
python run.py config=dann dataset.src_task=Cl dataset.tgt_task=Rw

python run.py config=dann dataset.src_task=Pr dataset.tgt_task=Ar
python run.py config=dann dataset.src_task=Pr dataset.tgt_task=Cl
python run.py config=dann dataset.src_task=Pr dataset.tgt_task=Rw

python run.py config=dann dataset.src_task=Rw dataset.tgt_task=Ar
python run.py config=dann dataset.src_task=Rw dataset.tgt_task=Cl
python run.py config=dann dataset.src_task=Rw dataset.tgt_task=Pr