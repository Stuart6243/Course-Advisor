import importlib.util
from pathlib import Path
import unittest

MODULE_PATH = Path('columbia_engineering_courses/scrape_columbia_courses.py')
spec = importlib.util.spec_from_file_location('scrape_columbia_courses', MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class TestScraperHelpers(unittest.TestCase):
    def test_parse_title_line_normal(self):
        code, title, points, issues = mod.parse_title_line('COMS W1004 PROGRAMMING IN JAVA. 3.00 points.')
        self.assertEqual(code, 'COMS W1004')
        self.assertEqual(title, 'PROGRAMMING IN JAVA')
        self.assertEqual(points, '3.00 points')
        self.assertEqual(issues, [])

    def test_parse_title_line_range(self):
        code, title, points, issues = mod.parse_title_line("ENGI E4990 ADVANCED MASTER'S RESEARCH. 0.00-6.00 points.")
        self.assertEqual(code, 'ENGI E4990')
        self.assertEqual(points, '0.00-6.00 points')
        self.assertEqual(issues, [])

    def test_parse_title_line_points_space_dot(self):
        code, title, points, issues = mod.parse_title_line('AEME E4304 Turbomachinery. 3.00 points .')
        self.assertEqual(code, 'AEME E4304')
        self.assertEqual(title, 'Turbomachinery')
        self.assertEqual(points, '3.00 points')
        self.assertEqual(issues, [])

    def test_parse_enrollment(self):
        current, cap = mod.parse_enrollment('38/40')
        self.assertEqual(current, 38)
        self.assertEqual(cap, 40)

    def test_dedup_key_stable(self):
        k1 = mod.build_dedup_key('2025-2026', 'COMS W1004', 'PROGRAMMING IN JAVA', 'https://bulletin.columbia.edu/columbia-engineering/academic-departments-programs/computer-science/')
        k2 = mod.build_dedup_key('2025-2026', 'COMS W1004', 'PROGRAMMING IN JAVA', 'https://bulletin.columbia.edu/columbia-engineering/academic-departments-programs/computer-science/')
        self.assertEqual(k1, k2)

    def test_parse_section_rows(self):
        from bs4 import BeautifulSoup

        html = '''
        <div class="courseblock">
          <div class="desc_sched">
            <table class="scheduletbl unifyBorder">
              <tr><td><strong>Fall 2025: COMS W1004</strong></td></tr>
              <tr>
                <th>COURSE NUMBER</th><th>SECTION/CALL NUMBER</th><th>TIMES/LOCATION</th><th>INSTRUCTOR</th><th>POINTS</th><th>ENROLLMENT</th>
              </tr>
              <tr>
                <td>COMS 1004</td><td>001/12794</td><td>M W 2:40pm - 3:55pm\n309 Havemeyer Hall</td><td>Paul Blaer</td><td>3.00</td><td>268/320</td>
              </tr>
            </table>
          </div>
        </div>
        '''
        soup = BeautifulSoup(html, 'html.parser')
        block = soup.select_one('.courseblock')
        sections, issues = mod.parse_section_rows(block)

        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0]['term'], 'Fall 2025')
        self.assertEqual(sections[0]['section_call_number'], '001/12794')
        self.assertEqual(sections[0]['location'], '309 Havemeyer Hall')
        self.assertEqual(issues, [])

    def test_parse_course_block_reclassifies_prereq_and_description(self):
        from bs4 import BeautifulSoup

        html = '''
        <div class="courseblock">
          <p class="courseblocktitle"><strong>AERO E4101 Aerospace Structures. <em>3.00 points</em>.</strong></p>
          <p class="courseblockdesc">Prerequisites: ( ENME E3105 and ENME E3113 )</p>
          <p>Analysis of aerospace structures including bending, shear, torsion, and stability methods in real aircraft components.</p>
          <p>Lect: 3.</p>
        </div>
        '''
        soup = BeautifulSoup(html, 'html.parser')
        block = soup.select_one('.courseblock')

        rec = mod.parse_course_block(
            block=block,
            year='2025-2026',
            source_url='https://bulletin.columbia.edu/columbia-engineering/academic-departments-programs/civil-engineering-engineering-mechanics/',
            source_page_title='test',
            department='test',
        )

        self.assertEqual(rec['course_code'], 'AERO E4101')
        self.assertTrue(rec['description'].startswith('Analysis of aerospace structures'))
        self.assertEqual(rec['description_source'], 'extra_paragraph')
        self.assertTrue(rec['prerequisites_text'].startswith('Prerequisites:'))
        self.assertEqual(rec['prerequisites_source'], 'courseblockdesc')
        self.assertEqual(rec['notes_text'], 'Lect: 3.')

    def test_parse_course_block_splits_prereq_sentence_tail(self):
        from bs4 import BeautifulSoup

        html = '''
        <div class="courseblock">
          <p class="courseblocktitle"><strong>APAM E3899 Research Training. <em>3.00 points</em>.</strong></p>
          <p class="courseblockdesc">Prerequisites: Instructor's permission. Research training course in preparation for laboratory-related research.</p>
        </div>
        '''
        soup = BeautifulSoup(html, 'html.parser')
        block = soup.select_one('.courseblock')

        rec = mod.parse_course_block(
            block=block,
            year='2025-2026',
            source_url='https://bulletin.columbia.edu/columbia-engineering/academic-departments-programs/applied-physics-applied-mathematics/',
            source_page_title='test',
            department='test',
        )

        self.assertTrue(rec['prerequisites_text'].startswith('Prerequisites: Instructor'))
        self.assertTrue(rec['description'].startswith('Research training course'))

    def test_merge_course_records_prefers_non_prereq_description(self):
        base = {
            'description': 'Prerequisites: ENME E3105',
            'description_source': 'courseblockdesc',
            'prerequisites_text': '',
            'prerequisites_source': 'none',
            'notes_text': '',
            'sections': [],
            'parse_warnings': [],
            'needs_review': False,
        }
        new = {
            'description': 'Actual course description text with learning objectives and topic coverage.',
            'description_source': 'extra_paragraph',
            'prerequisites_text': 'Prerequisites: ENME E3105',
            'prerequisites_source': 'courseblockdesc',
            'notes_text': 'Lect: 3.',
            'sections': [],
            'parse_warnings': [],
            'needs_review': False,
        }

        merged = mod.merge_course_records(base, new)
        self.assertTrue(merged['description'].startswith('Actual course description'))
        self.assertEqual(merged['description_source'], 'extra_paragraph')
        self.assertTrue(merged['prerequisites_text'].startswith('Prerequisites:'))


if __name__ == '__main__':
    unittest.main()
