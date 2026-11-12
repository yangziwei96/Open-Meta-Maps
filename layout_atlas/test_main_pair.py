"""Fixtures for main-pair selection (formula overlap, currency tiers, neighbour links)."""
from core import currency
import main_pair as mp


def _model(mets, reactions):
    return dict(
        id='t',
        metabolites=[dict(id=m, name=n, compartment=c, chemicalFormula=f)
                     for m, n, c, f in mets],
        reactions=[dict(id=rid, metabolites=stoich, lower_bound=0, upper_bound=1000)
                   for rid, stoich in reactions],
    )


def test_prefers_carbon_carrier_over_cofactor_pair():
    # Pyruvate + CoA + NAD → acetyl-CoA + NADH + CO2: carbon transfer beats CoA→acetyl-CoA.
    model = _model(
        [
            ('pyr_c', 'Pyruvate', 'c', 'C3H3O3'),
            ('coa_c', 'CoA', 'c', 'C21H32N7O16P3S'),
            ('nad_c', 'NAD', 'c', 'C21H26N7O14P2'),
            ('accoa_c', 'Acetyl-CoA', 'c', 'C23H34N7O17P3S'),
            ('nadh_c', 'NADH', 'c', 'C21H27N7O14P2'),
            ('co2_c', 'CO2', 'c', 'CO2'),
        ],
        [('PDH', {'pyr_c': -1, 'coa_c': -1, 'nad_c': -1, 'accoa_c': 1, 'nadh_c': 1, 'co2_c': 1})],
    )
    cur = currency(model)
    pairs = mp.choose_all(model, cur=cur)
    assert pairs['PDH'] == ('pyr_c', 'accoa_c')


def test_currency_never_wins_while_non_currency_exists():
    model = _model(
        [
            ('a_c', 'A', 'c', 'C6H12O6'),
            ('b_c', 'B', 'c', 'C6H11O9P'),
            ('atp_c', 'ATP', 'c', 'C10H12N5O13P3'),
            ('adp_c', 'ADP', 'c', 'C10H12N5O10P2'),
            ('h_c', 'H+', 'c', 'H'),
        ],
        [('HEX', {'a_c': -1, 'atp_c': -1, 'b_c': 1, 'adp_c': 1, 'h_c': 1})],
    )
    pairs = mp.choose_all(model, cur=currency(model))
    assert pairs['HEX'] == ('a_c', 'b_c')


def test_equal_chemistry_prefers_pair_with_drawn_neighbours():
    # When two carbon pairs score equally, the end already linked to another drawn reaction wins.
    model = _model(
        [
            ('accoa_c', 'Acetyl-CoA', 'c', 'C23H34N7O17P3S'),
            ('oaa_c', 'Oxaloacetate', 'c', 'C4H2O5'),
            ('cit_c', 'Citrate', 'c', 'C6H5O7'),
            ('coa_c', 'CoA', 'c', 'C21H32N7O16P3S'),
            ('h2o_c', 'H2O', 'c', 'H2O'),
            ('mal_c', 'Malate', 'c', 'C4H4O5'),
        ],
        [
            ('CS', {'accoa_c': -1, 'oaa_c': -1, 'h2o_c': -1, 'cit_c': 1, 'coa_c': 1}),
            ('MDH', {'mal_c': -1, 'oaa_c': 1}),
        ],
    )
    cur = currency(model)
    scene = dict(
        participants={
            ('CS', 'accoa_c'): 'm:accoa_c', ('CS', 'oaa_c'): 'm:oaa_c', ('CS', 'cit_c'): 'm:cit_c',
            ('CS', 'coa_c'): 'side:CS:coa_c', ('CS', 'h2o_c'): 'side:CS:h2o_c',
            ('MDH', 'mal_c'): 'm:mal_c', ('MDH', 'oaa_c'): 'm:oaa_c',
        },
        copies={},
        positions={},
    )
    pairs = mp.choose_all(model, scene, cur)
    assert pairs['CS'] == ('oaa_c', 'cit_c')


def test_transport_same_compound_when_no_conversion():
    model = _model(
        [
            ('pi_c', 'Phosphate', 'c', 'HO4P'),
            ('pi_e', 'Phosphate', 'e', 'HO4P'),
            ('h_c', 'H+', 'c', 'H'),
            ('h_e', 'H+', 'e', 'H'),
        ],
        [('PIt', {'pi_e': -1, 'h_e': -1, 'pi_c': 1, 'h_c': 1})],
    )
    pairs = mp.choose_all(model, cur=currency(model))
    assert pairs['PIt'] == ('pi_e', 'pi_c')


def test_all_currency_falls_back_to_chemistry():
    model = _model(
        [
            ('atp_c', 'ATP', 'c', 'C10H12N5O13P3'),
            ('h2o_c', 'H2O', 'c', 'H2O'),
            ('adp_c', 'ADP', 'c', 'C10H12N5O10P2'),
            ('pi_c', 'Phosphate', 'c', 'HO4P'),
            ('h_c', 'H+', 'c', 'H'),
        ],
        [('ATPM', {'atp_c': -1, 'h2o_c': -1, 'adp_c': 1, 'pi_c': 1, 'h_c': 1})],
    )
    pairs = mp.choose_all(model, cur=currency(model))
    assert pairs['ATPM'] == ('atp_c', 'adp_c')


def test_side_metabolites_lists_non_main():
    reaction = dict(id='R', metabolites={'a_c': -1, 'atp_c': -1, 'b_c': 1, 'adp_c': 1})
    consumed, produced = mp.side_metabolites(reaction, 'a_c', 'b_c')
    assert consumed == ['atp_c'] and produced == ['adp_c']


def test_choose_all_deterministic():
    model = _model(
        [
            ('a_c', 'A', 'c', 'C4H4O5'),
            ('b_c', 'B', 'c', 'C4H2O5'),
            ('c_c', 'C', 'c', 'C4H4O5'),
            ('d_c', 'D', 'c', 'C4H2O5'),
        ],
        [('R1', {'a_c': -1, 'b_c': 1}), ('R2', {'c_c': -1, 'd_c': 1})],
    )
    a = mp.choose_all(model)
    b = mp.choose_all(model)
    assert a == b == {'R1': ('a_c', 'b_c'), 'R2': ('c_c', 'd_c')}
