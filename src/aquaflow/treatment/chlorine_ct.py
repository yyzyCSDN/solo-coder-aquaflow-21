from dataclasses import dataclass
@dataclass(frozen=True)
class Segment:
    minutes:float; residual:float; baffling:float; temperature_c:float
def effective_ct(rows:list[Segment])->float:
    total=0.0
    for r in rows:
        if r.minutes<=0 or r.residual<0 or not 0<r.baffling<=1: raise ValueError("invalid")
        tf=max(.55,min(1.25,1+(r.temperature_c-20)*.015))
        total+=r.minutes*r.residual*r.baffling*tf
    return total
def compliant(rows,required): return required>0 and effective_ct(rows)>=required
