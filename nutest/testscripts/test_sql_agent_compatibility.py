"""Review9/10: valid HANA SELECT grammar must preserve object authorization."""
import os
import uuid
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine, event

from hana_ai.agents.hana_sql_agent import _ReadOnlySQLDatabase, SQLDatabase


VALID_SELECTS = [
    'SELECT CAST(ID AS INTEGER) FROM {table}',
    'SELECT CAST(ID AS DECIMAL(10, 2)) FROM {table}',
    'SELECT CAST(TO_VARCHAR(ID) AS NVARCHAR(20)) FROM {table}',
    "SELECT CAST('1' AS DECIMAL(10, 2)) FROM {table}",
    "SELECT CAST('O''Reilly; SELECT' AS NVARCHAR(40)) FROM {table}",
    'SELECT CAST((SELECT MAX(ID) FROM {table}) AS INTEGER) FROM {table}',
    'SELECT TO_VARCHAR(ID) FROM {table}',
    'SELECT ID FROM {table} WHERE ID = ANY (SELECT ID FROM {table})',
    'SELECT ID FROM {table} WHERE ID = SOME (SELECT ID FROM {table})',
    'SELECT ID FROM {table} WITH HINT (NO_INLINE)',
    'SELECT ID FROM {table} WITH HINT (NO_INLINE, CS_JOIN)',
    'SELECT ID FROM {table} WITH HINT (MAX_CONCURRENCY(1))',
    'SELECT ID FROM {table} PARTITION (1, 2)',
    'SELECT A.ID FROM {table} PARTITION (1, 2) AS A',
    'SELECT ID FROM {table} TABLESAMPLE SYSTEM (100)',
    'SELECT A.ID FROM {table} AS A TABLESAMPLE BERNOULLI (100)',
    'SELECT B.ID FROM {table} A CROSS JOIN LATERAL '
    '(SELECT ID FROM {table} C WHERE C.ID = A.ID) B',
    'SELECT B.ID FROM {table} A CROSS JOIN LATERAL '
    '(WITH X AS (SELECT ID FROM {table}) SELECT ID FROM X WHERE X.ID = A.ID) B',
    'SELECT A.ID FROM {table} A JOIN {table} B '
    'ON LEFT(TO_VARCHAR(A.ID),1)=LEFT(TO_VARCHAR(B.ID),1)',
    'SELECT ID FROM {table} WITH HINT (NO_INLINE);',
    'select ID from {table} with hint (no_inline)',
    'SELECT A.ID FROM {table} A JOIN {table} B ON A.ID=B.ID WITH HINT (NO_INLINE)',
    'SELECT A.ID FROM {table} A LEFT JOIN {table} B '
    'ON (A.ID=B.ID) WITH HINT (NO_INLINE)',
    'SELECT ID FROM {table} WHERE (ID > 0) ORDER BY ID WITH HINT (NO_INLINE)',
    'WITH X AS (SELECT ID FROM {table}) SELECT ID FROM X WITH HINT (NO_INLINE)',
    'SELECT ID FROM (SELECT ID FROM {table}) X WITH HINT (NO_INLINE)',
    'SELECT ID FROM {table} UNION SELECT ID FROM {table} WITH HINT (NO_INLINE)',
    'SELECT ID FROM {table} WITH HINT (MAX_CONCURRENCY(1), NO_INLINE)',
    "SELECT CAST('WITH HINT (, JOIN SECRET_TABLE)' AS NVARCHAR(40)) FROM {table} "
    'WITH HINT (NO_INLINE)',
]


INVALID_HINT_SELECTS = [
    pytest.param('SELECT * FROM {table} WITH HINT (NO_INLINE), {secret}', id='comma'),
    pytest.param('SELECT * FROM {table} WITH HINT (NO_INLINE) '
                 'JOIN {secret} S ON 1=1', id='join'),
    pytest.param('SELECT * FROM {table} WITH HINT (NO_INLINE) '
                 'CROSS JOIN {secret} S', id='cross-join'),
    pytest.param('SELECT * FROM {table} WITH HINT (NO_INLINE) '
                 'LEFT JOIN {secret} S ON 1=1', id='left-join'),
    pytest.param('SELECT * FROM {table} WITH HINT (NO_INLINE) '
                 'INNER JOIN {secret} S ON 1=1', id='inner-join'),
    pytest.param('SELECT * FROM (SELECT * FROM {table} WITH HINT (NO_INLINE) '
                 'JOIN {secret} S ON 1=1) X', id='nested-join'),
    pytest.param('WITH X AS (SELECT * FROM {table} WITH HINT (NO_INLINE) '
                 'JOIN {secret} S ON 1=1) SELECT * FROM X', id='cte-join'),
    pytest.param('SELECT * FROM (SELECT * FROM {table} WITH HINT (NO_INLINE)) X',
                 id='nested-terminal-hint'),
    pytest.param('WITH X AS (SELECT * FROM {table} WITH HINT (NO_INLINE)) '
                 'SELECT * FROM X', id='cte-terminal-hint'),
    pytest.param('SELECT (SELECT MAX(ID) FROM {table} WITH HINT (NO_INLINE)) '
                 'FROM {table}', id='scalar-subquery'),
    pytest.param('SELECT CAST((SELECT MAX(ID) FROM {table} WITH HINT (NO_INLINE)) '
                 'AS INTEGER) FROM {table}', id='cast-subquery'),
    pytest.param('SELECT * FROM {table} WHERE (ID > 0 WITH HINT (NO_INLINE))',
                 id='predicate'),
    pytest.param('SELECT * FROM {table} WITH HINT (NO_INLINE) WHERE ID > 0',
                 id='before-where'),
    pytest.param('SELECT * FROM {table} WITH HINT (NO_INLINE) ORDER BY ID',
                 id='before-order'),
    pytest.param('SELECT * FROM {table} WITH HINT (NO_INLINE) LIMIT 1',
                 id='before-limit'),
    pytest.param('SELECT * FROM {table} WITH HINT (NO_INLINE) '
                 'UNION SELECT * FROM {table}', id='before-union'),
    pytest.param('SELECT * FROM {table} WITH HINT (NO_INLINE) '
                 'JOIN {table} A ON 1=1', id='before-allowed-join'),
    pytest.param('SELECT * FROM {table} WITH HINT (NO_INLINE) WITH HINT (CS_JOIN)',
                 id='repeated-hint'),
    pytest.param('SELECT 1 WITH HINT (NO_INLINE) FROM {table}', id='before-from'),
    pytest.param('SELECT * FROM {table} WITH HINT (NO_INLINE) '
                 'JOIN {secret} S ON 1=1 WITH HINT (CS_JOIN)', id='join-between-hints'),
]


@pytest.fixture
def scoped_database():
    engine = create_engine('sqlite://')
    with engine.begin() as connection:
        connection.exec_driver_sql('CREATE TABLE ALLOWED_TABLE (ID INTEGER)')
        connection.exec_driver_sql('CREATE TABLE SECRET_TABLE (ID INTEGER)')
        connection.exec_driver_sql('INSERT INTO ALLOWED_TABLE VALUES (1)')
    database = _ReadOnlySQLDatabase(engine, allowed_tables=['ALLOWED_TABLE'])
    executed = []
    event.listen(engine, 'before_cursor_execute', lambda *args: executed.append(args[2]))
    yield database, executed
    engine.dispose()


@pytest.mark.parametrize('template', VALID_SELECTS)
def test_hana_grammar_reaches_executor(scoped_database, monkeypatch, template):
    database, _ = scoped_database
    # SQLite cannot execute HANA-specific syntax. Live tests below compare
    # execution against HANA itself; here we isolate the authorization boundary.
    monkeypatch.setattr(SQLDatabase, 'run', lambda self, command: command)
    statement = template.format(table='ALLOWED_TABLE')
    assert database.run(statement) == statement


@pytest.mark.parametrize('statement', [
    'SELECT CAST((SELECT ID FROM SECRET_TABLE) AS INTEGER) FROM ALLOWED_TABLE',
    'SELECT CAST(1+SECRET_SCHEMA.GET_SECRET() AS DECIMAL(10,2)) FROM ALLOWED_TABLE',
    'SELECT CAST(1+SECRET_SEQ.NEXTVAL AS INTEGER) FROM ALLOWED_TABLE',
    'SELECT CAST(ID AS SECRET_SCHEMA.SECRET_TYPE) FROM ALLOWED_TABLE',
    'SELECT CAST(ID AS DECIMAL(GET_SECRET(),2)) FROM ALLOWED_TABLE',
    'SELECT TO_VARCHAR(GET_SECRET()) FROM ALLOWED_TABLE',
    'SELECT ID FROM ALLOWED_TABLE WHERE ID = ANY (SELECT ID FROM SECRET_TABLE)',
    'SELECT ID FROM ALLOWED_TABLE WHERE ID = SOME (SELECT SECRET_SEQ.NEXTVAL FROM ALLOWED_TABLE)',
    'SELECT * FROM ALLOWED_TABLE CROSS JOIN LATERAL (SELECT ID FROM SECRET_TABLE) X',
    'SELECT * FROM LATERAL SECRET_TABLE_FUNCTION()',
    'SELECT * FROM ALLOWED_TABLE CROSS JOIN LATERAL '
    '(WITH SECRET_TABLE AS (SELECT ID FROM ALLOWED_TABLE) SELECT * FROM SECRET_TABLE) X '
    'JOIN SECRET_TABLE S ON X.ID=S.ID',
    'SELECT ID FROM SECRET_TABLE PARTITION (1)',
    'SELECT ID FROM ALLOWED_TABLE PARTITION (GET_SECRET())',
    'SELECT ID FROM ALLOWED_TABLE PARTITION (1+SECRET_SEQ.NEXTVAL)',
    'SELECT ID FROM SECRET_TABLE TABLESAMPLE SYSTEM (100)',
    'SELECT ID FROM ALLOWED_TABLE TABLESAMPLE SYSTEM (GET_SECRET())',
    'SELECT ID FROM ALLOWED_TABLE TABLESAMPLE SYSTEM (1+SECRET_SEQ.NEXTVAL)',
    'SELECT ID FROM ALLOWED_TABLE WITH HINT (SECRET_SCHEMA.GET_SECRET())',
    'SELECT ID FROM ALLOWED_TABLE WITH HINT (NO_INLINE(GET_SECRET()))',
    'SELECT ID FROM ALLOWED_TABLE WITH HINT ((SELECT ID FROM SECRET_TABLE))',
    'SELECT ID FROM ALLOWED_TABLE WITH HINT (NO_INLINE) UNION SELECT ID FROM SECRET_TABLE',
    'SELECT ID FROM ALLOWED_TABLE WITH HINT (NO_INLINE); DELETE FROM ALLOWED_TABLE',
])
def test_compatibility_syntax_cannot_hide_unauthorized_objects(scoped_database, statement):
    database, executed = scoped_database
    with pytest.raises(ValueError):
        database.run(statement)
    assert not executed


def test_comma_after_join_condition_cannot_hide_table(scoped_database):
    database, executed = scoped_database
    with pytest.raises(ValueError):
        database.run('SELECT S.ID FROM ALLOWED_TABLE A JOIN ALLOWED_TABLE B '
                     'ON A.ID=B.ID, SECRET_TABLE S')
    assert not executed


@pytest.mark.parametrize('template', INVALID_HINT_SELECTS)
def test_invalid_hint_placement_never_reaches_executor(scoped_database, monkeypatch, template):
    database, executed = scoped_database
    # Simulate a permissive downstream executor: rejection must come from our
    # guard, not SQLite/HANA rejecting the malformed SQL as a syntax error.
    executor = Mock(return_value='unexpected execution')
    monkeypatch.setattr(SQLDatabase, 'run', executor)
    statement = template.format(table='ALLOWED_TABLE', secret='SECRET_TABLE')
    with pytest.raises(ValueError, match='WITH HINT must end the outer SELECT statement'):
        database.run(statement)
    executor.assert_not_called()
    assert not executed


@pytest.fixture(scope='module')
def live_hana_database():
    userkey = os.environ.get('HANA_USERKEY')
    if not userkey:
        pytest.skip('HANA_USERKEY is required for live SQL compatibility tests')
    from hana_ml.dataframe import ConnectionContext
    cc = ConnectionContext(userkey=userkey, encrypt=True, sslValidateCertificate=True)
    cursor = cc.connection.cursor()
    schema = cc.get_current_schema()
    name = 'R9_COMPAT_' + uuid.uuid4().hex[:12].upper()
    quoted_schema = '"' + schema.replace('"', '""') + '"'
    table = f'{quoted_schema}."{name}"'
    engine = None
    created = False
    try:
        cursor.execute(f'CREATE COLUMN TABLE {table} (ID INTEGER) PARTITION BY HASH (ID) PARTITIONS 2')
        created = True
        cursor.executemany(f'INSERT INTO {table} VALUES (?)', [(n,) for n in range(1, 5)])
        cc.connection.commit()
        engine = cc.to_sqlalchemy()
        database = _ReadOnlySQLDatabase(
            engine, schema=schema, include_tables=[name],
            allowed_schema=quoted_schema, allowed_tables=[name],
        )
        yield database, cursor, table
    finally:
        if engine is not None:
            engine.dispose()
        try:
            if created:
                cursor.execute(f'DROP TABLE {table}')
        finally:
            cursor.close()
            cc.close()


@pytest.mark.parametrize('template', VALID_SELECTS)
def test_valid_select_matches_direct_hana_execution(live_hana_database, template):
    database, cursor, table = live_hana_database
    statement = template.format(table=table)
    cursor.execute(statement)
    expected = sorted(tuple(row) for row in cursor.fetchall())
    with database.run(statement, fetch='cursor') as result:
        actual = sorted(tuple(row) for row in result.fetchall())
    assert actual == expected
    assert len(actual) == 4


def test_live_comma_after_join_is_rejected_before_execute(live_hana_database):
    database, _, table = live_hana_database
    executed = []
    def record_execution(*args):
        executed.append(args[2])
    event.listen(database._engine, 'before_cursor_execute', record_execution)
    try:
        with pytest.raises(ValueError, match='comma-separated'):
            database.run(f'SELECT S.DUMMY FROM {table} A JOIN {table} B '
                         'ON A.ID=B.ID, SYS.DUMMY S')
        assert not executed
    finally:
        event.remove(database._engine, 'before_cursor_execute', record_execution)


@pytest.mark.parametrize('template', INVALID_HINT_SELECTS)
def test_live_invalid_hint_placement_is_rejected_before_execute(live_hana_database, template):
    database, _, table = live_hana_database
    executed = []

    def record_execution(*args):
        executed.append(args[2])

    event.listen(database._engine, 'before_cursor_execute', record_execution)
    try:
        with pytest.raises(ValueError, match='WITH HINT must end the outer SELECT statement'):
            database.run(template.format(table=table, secret='SYS.DUMMY'))
        assert not executed
    finally:
        event.remove(database._engine, 'before_cursor_execute', record_execution)
